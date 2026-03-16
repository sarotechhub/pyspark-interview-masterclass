"""
Q17: Deduplicate across two DataFrames and combine (union distinct).

Scenario: Merge two data sources that may contain duplicate records.
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - union(): O(n+m) — concatenates rows (no cost)
#   - dropDuplicates(): O((n+m)*log(n+m)) — hash dedup with sort
#   - Total: O((n+m)*log(n+m)) — dominated by dedup
#
# Shuffle Operations:
#   - union(): NO SHUFFLE (just concatenation, no repartition)
#   - dropDuplicates(): FULL SHUFFLE (hash-based dedup)
#   - unionByName(): NO SHUFFLE (concat by column name)
#
# Performance Tips:
#   - union() vs unionByName(): select which columns first
#   - dropDuplicates() uses ALL columns for comparison
#   - dropDuplicates(subset=[...]) for specific columns (faster)
#   - Broadcast smaller table if joining instead
#   - Cache after union before multiple operations
#   - Consider: distinct() as alias for dropDuplicates()
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Q17_UnionDistinct") \
    .master("local[*]") \
    .getOrCreate()

# ============================================
# SAMPLE DATA
# ============================================
df_source1 = spark.createDataFrame([
    (1, "Alice", 30),
    (2, "Bob", 25),
    (3, "Carol", 35),
], ["id", "name", "age"])

df_source2 = spark.createDataFrame([
    (2, "Bob", 25),      # Exact duplicate
    (3, "Carol", 35),    # Exact duplicate
    (4, "Dave", 40),     # New record
], ["id", "name", "age"])

print("=" * 60)
print("SOURCE 1:")
print("=" * 60)
df_source1.show()

print("=" * 60)
print("SOURCE 2:")
print("=" * 60)
df_source2.show()

# ============================================
# SOLUTION 1: Union all, then deduplicate
# ============================================

# Step 1: Use union() to combine both DataFrames
#         union() keeps ALL rows including duplicates
combined = df_source1.union(df_source2)

print("\n" + "=" * 60)
print("UNION (includes duplicates):")
print("=" * 60)
combined.show()

# Step 2: Remove duplicate rows using dropDuplicates()
result_all_cols = combined.dropDuplicates()

print("\n" + "=" * 60)
print("OUTPUT: Union + Deduplicate (all columns):")
print("=" * 60)
result_all_cols.show()

# ============================================
# SOLUTION 2: Deduplicate on specific columns
# ============================================

# Deduplicate based only on 'id' column
# (keeps first occurrence if different values for other columns)
result_by_id = combined.dropDuplicates(["id"])

print("\n" + "=" * 60)
print("Deduplicate by ID column only:")
print("=" * 60)
result_by_id.show()

# ============================================
# SOLUTION 3: Using unionByName
# ============================================

# unionByName matches columns by name (safer than positional union)
result_union_by_name = df_source1.unionByName(df_source2).dropDuplicates()

print("\n" + "=" * 60)
print("Using unionByName (column-name safe):")
print("=" * 60)
result_union_by_name.show()

# ============================================
# ADVANCED: Handle different schemas
# ============================================

# Source 3 with additional column
df_source3 = spark.createDataFrame([
    (5, "Eve", 28, "active"),
], ["id", "name", "age", "status"])

# Add missing column to make schemas compatible
df_source1_compat = df_source1.withColumn("status", F.lit(None))
df_source2_compat = df_source2.withColumn("status", F.lit(None))

result_different_schema = df_source1_compat \
    .unionByName(df_source2_compat) \
    .unionByName(df_source3) \
    .dropDuplicates(["id"])

print("\n" + "=" * 60)
print("UNION with different schemas (adding status column):")
print("=" * 60)
result_different_schema.show()

# ============================================
# ADVANCED: Window function to detect duplicates
# ============================================

from pyspark.sql.window import Window

combined_with_dup_flag = combined \
    .withColumn("row_num", F.row_number().over(Window.partitionBy("id").orderBy(F.lit(1)))) \
    .withColumn("is_duplicate", F.col("row_num") > 1)

print("\n" + "=" * 60)
print("Mark duplicates before deduplication:")
print("=" * 60)
combined_with_dup_flag.show()

# Get duplicate summary
duplicate_summary = combined_with_dup_flag \
    .groupBy("id", "name", "age") \
    .agg(F.count("*").alias("occurrence_count"))

print("\n" + "=" * 60)
print("Duplicate Summary (which records appeared how many times):")
print("=" * 60)
duplicate_summary.show()

# ============================================
# BEST PRACTICE: Compare before and after
# ============================================

print("\n" + "=" * 60)
print("COMPARISON:")
print("=" * 60)
print(f"Source 1 rows: {df_source1.count()}")
print(f"Source 2 rows: {df_source2.count()}")
print(f"Combined rows (with duplicates): {combined.count()}")
print(f"After deduplication: {result_all_cols.count()}")
print(f"Duplicates removed: {combined.count() - result_all_cols.count()}")

# ============================================
# EXPECTED OUTPUT:
# ============================================
# OUTPUT: Union + Deduplicate
# +---+-----+---+
# | id| name|age|
# +---+-----+---+
# |  1|Alice| 30|
# |  2|  Bob| 25|
# |  3|Carol| 35|
# |  4| Dave| 40|
# +---+-----+---+
#
# Notes:
# - Original records 2 and 3 appeared in both sources (duplicates)
# - Record 1 is unique to Source 1
# - Record 4 is unique to Source 2
# - Final result has 4 unique records

spark.stop()
