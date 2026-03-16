"""
Q18: Perform a fuzzy/approximate join on names using similarity.

Scenario: Two systems have customer names with slight variations. Match them.
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - crossJoin(): O(n*m) — Cartesian product (EXPENSIVE!)
#   - levenshtein(): O(k^2) — k = string length per comparison
#   - Total: O(n*m*k^2) — quadratic in table sizes!
#   - For 1000×1000 with k=20: 20-40 billion operations
#
# Shuffle Operations:
#   - crossJoin(): FULL SHUFFLE (broadcast join cartesian product)
#   - Very expensive operation
#
# Performance Tips:
#   - NEVER use on large tables (100M+ rows)
#   - crossJoin() only for small tables (<10K rows)
#   - Pre-filter both tables aggressively (by length, prefix, etc.)
#   - Block join: match by first letters before fuzzy
#   - Consider: Soundex/Metaphone for pronunciation matching
#   - Levenshtein distance threshold (typically 1-3 edits)
#   - Alternative: ML-based matching (more complex but needed for scale)
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Q18_FuzzyJoin") \
    .master("local[*]") \
    .getOrCreate()

# ============================================
# SAMPLE DATA
# ============================================
df1 = spark.createDataFrame([
    (1, "John Smith"),
    (2, "Alice Johnson"),
    (3, "Robert Williams"),
], ["id1", "name1"])

df2 = spark.createDataFrame([
    (101, "Jon Smith"),        # Typo - missing 'h'
    (102, "Alice Johnston"),   # Typo - 'Johnston' instead of 'Johnson'
    (103, "Bob Williams"),     # Shortened version
    (104, "Carol Davis"),      # No match in df1
], ["id2", "name2"])

print("=" * 60)
print("TABLE 1 (Source A):")
print("=" * 60)
df1.show(truncate=False)

print("=" * 60)
print("TABLE 2 (Source B):")
print("=" * 60)
df2.show(truncate=False)

# ============================================
# SOLUTION: Using Levenshtein Distance
# ============================================

# Step 1: Cross join both tables (all combinations)
result = df1.crossJoin(df2)

# Step 2: Calculate Levenshtein distance using built-in function
#         levenshtein(str1, str2) returns number of edits needed
result = result.withColumn(
    "distance",
    F.levenshtein(F.col("name1"), F.col("name2"))
)

# Step 3: Filter for good matches (distance <= 3 is typical)
result_filtered = result.filter(F.col("distance") <= 3) \
    .orderBy("distance") \
    .select("id1", "name1", "id2", "name2", "distance")

print("\n" + "=" * 60)
print("FUZZY JOIN RESULTS (Levenshtein distance <= 3):")
print("=" * 60)
result_filtered.show(truncate=False)

# ============================================
# ADVANCED: Similarity percentage
# ============================================

# Calculate similarity as a percentage
max_length = F.greatest(F.length("name1"), F.length("name2"))
result_sim = df1.crossJoin(df2) \
    .withColumn("distance", F.levenshtein(F.col("name1"), F.col("name2"))) \
    .withColumn("similarity_pct",
        F.round((1 - F.col("distance") / max_length) * 100, 2)
    ) \
    .filter(F.col("similarity_pct") >= 80) \
    .orderBy(F.desc("similarity_pct"))

print("\n" + "=" * 60)
print("WITH SIMILARITY PERCENTAGE (>= 80%):")
print("=" * 60)
result_sim.select(
    "id1", "name1", "id2", "name2", 
    "distance", "similarity_pct"
).show(truncate=False)

# ============================================
# ALTERNATIVE: Using Soundex (phonetic matching)
# ============================================

# Soundex encodes names by sound
result_soundex = df1.crossJoin(df2) \
    .withColumn("soundex1", F.soundex("name1")) \
    .withColumn("soundex2", F.soundex("name2")) \
    .withColumn("soundex_match",
        F.col("soundex1") == F.col("soundex2")
    ) \
    .filter(F.col("soundex_match") == True)

print("\n" + "=" * 60)
print("PHONETIC MATCHING (Using Soundex):")
print("=" * 60)
result_soundex.select(
    "id1", "name1", "soundex1",
    "id2", "name2", "soundex2"
).show(truncate=False)

# ============================================
# BEST MATCH PER RECORD
# ============================================

from pyspark.sql.window import Window

# Find the best match for each name1
window_best = Window.partitionBy("id1").orderBy("distance")

best_matches = df1.crossJoin(df2) \
    .withColumn("distance", F.levenshtein(F.col("name1"), F.col("name2"))) \
    .withColumn("rank", F.row_number().over(window_best)) \
    .filter((F.col("rank") == 1) & (F.col("distance") <= 5))

print("\n" + "=" * 60)
print("BEST MATCH PER RECORD (single best match):")
print("=" * 60)
best_matches.select("id1", "name1", "id2", "name2", "distance").show(truncate=False)

# ============================================
# EXPECTED OUTPUT:
# ============================================
# RESULTS (distance <= 3):
# +---+-----------+----+--------+--------+
# |id1|name1      |id2  |name2   |distance|
# +---+-----------+----+--------+--------+
# | 1 |John Smith |101  |Jon Smith |1     |
# | 2 |Alice J... |102  |Alice J..|3     |
# | 3 |Robert W...|103  |Bob W... |7     | (filtered out)
# +---+-----------+----+--------+--------+

spark.stop()
