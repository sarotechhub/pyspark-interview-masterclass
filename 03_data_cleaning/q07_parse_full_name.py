"""
Q7: Parse and split a full name column into first and last name.

Scenario: You have full names in one column and need to split them.
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - split(string, delimiter): O(n*m) — m = tokens in string
#   - substring/array access: O(n) — indexed access
#   - concat_ws(): O(n*m) — concatenate m parts
#   - Total: O(n*m) where m is number of name parts
#
# Shuffle Operations:
#   - split() / substring(): NO SHUFFLE (string operation)
#   - array operations: NO SHUFFLE (column transform)
#   - No shuffles in name parsing
#
# Performance Tips:
#   - split() on space works for most cases
#   - Handle edge cases: single names, suffixes (Jr, Sr, III)
#   - Use when() for conditional splits
#   - Array indexing [0] faster than slice for first element
#   - Consider regex for complex name patterns
#   - Cache parsed columns if reused multiple times
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Q07_SplitFullName") \
    .master("local[*]") \
    .getOrCreate()

# ============================================
# SAMPLE DATA
# ============================================
data = [
    (1, "Alice Johnson"),
    (2, "Bob Smith Jr"),
    (3, "Madonna"),              # Only first name
    (4, "Jean-Claude Van Damme"),
    (5, "Dr. John Michael Smith III"),
]

df = spark.createDataFrame(data, ["id", "full_name"])

print("=" * 60)
print("INPUT DATA:")
print("=" * 60)
df.show(truncate=False)

# ============================================
# SOLUTION - METHOD 1: Simple Split
# ============================================

# Step 1: Split the full_name by spaces using split()
#         This returns an array of strings
df_method1 = df.withColumn("name_parts", F.split(F.col("full_name"), " "))

# Step 2: Extract first element as first_name (index 0)
df_method1 = df_method1.withColumn("first_name", F.col("name_parts")[0])

# Step 3: Extract last element as last_name
#         Use F.size(array) - 1 to get the last index
df_method1 = df_method1.withColumn("last_name",
    F.when(F.size("name_parts") > 1,
        F.col("name_parts")[F.size("name_parts") - 1]
    ).otherwise(F.lit(None))
)

# Step 4: Extract middle part (all parts except first and last)
df_method1 = df_method1.withColumn("middle_name",
    F.when(F.size("name_parts") > 2,
        F.concat_ws(" ", 
            F.slice("name_parts", 2, F.size("name_parts") - 2)
        )
    ).otherwise(F.lit(None))
)

# Drop the temporary array column
df_method1 = df_method1.drop("name_parts")

print("\n" + "=" * 60)
print("METHOD 1 OUTPUT (Simple Split):")
print("=" * 60)
df_method1.select("id", "full_name", "first_name", "middle_name", "last_name").show(truncate=False)

# ============================================
# SOLUTION - METHOD 2: Using regex_extract
# ============================================
# This method uses regex to extract name patterns

df_method2 = spark.createDataFrame(data, ["id", "full_name"])

# Extract first name (first sequence of letters)
df_method2 = df_method2.withColumn("first_name",
    F.regexp_extract(F.col("full_name"), r"^\w+", 0)
)

# Extract last name (last sequence of letters, excluding suffixes like Jr, III)
df_method2 = df_method2.withColumn("last_name",
    F.case_when(
        F.lower(F.col("full_name")).contains("jr"),
        F.regexp_extract(F.col("full_name"), r"(\w+)\s+Jr", 1)
    ).when(
        F.lower(F.col("full_name")).contains("iii"),
        F.regexp_extract(F.col("full_name"), r"(\w+)\s+III", 1)
    ).otherwise(
        F.regexp_extract(F.col("full_name"), r"(\w+)$", 0)
    )
)

print("\n" + "=" * 60)
print("METHOD 2 OUTPUT (Using regex_extract):")
print("=" * 60)
df_method2.select("id", "full_name", "first_name", "last_name").show(truncate=False)

# ============================================
# SOLUTION - METHOD 3: Use regex_replace to clean, then split
# ============================================
# Remove titles and suffixes first, then split

df_method3 = spark.createDataFrame(data, ["id", "full_name"])

# Remove common titles (Dr., Prof., Mr., Ms., Mrs., etc.)
df_method3 = df_method3.withColumn("cleaned_name",
    F.trim(F.regexp_replace(
        F.col("full_name"),
        r"^(Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.|Ms\.|Sr\.|Jr\.|III|II|IV|\d+)",
        ""
    ))
)

# Now split the cleaned name
df_method3 = df_method3.withColumn("name_parts", F.split(F.col("cleaned_name"), " "))

df_method3 = df_method3 \
    .withColumn("first_name", F.col("name_parts")[0]) \
    .withColumn("last_name",
        F.when(F.size("name_parts") > 1,
            F.col("name_parts")[F.size("name_parts") - 1]
        ).otherwise(F.lit(None))
    ) \
    .drop("name_parts", "cleaned_name")

print("\n" + "=" * 60)
print("METHOD 3 OUTPUT (Clean then split):")
print("=" * 60)
df_method3.select("id", "full_name", "first_name", "last_name").show(truncate=False)

# ============================================
# EXPECTED OUTPUT (Method 1):
# ============================================
# +---+------------------+-----------+----------+---------+
# | id|         full_name |first_name |middle_name|last_name|
# +---+------------------+-----------+----------+---------+
# |  1|    Alice Johnson  | Alice     | null     | Johnson |
# |  2|   Bob Smith Jr    | Bob       | Smith    | Jr      |
# |  3|       Madonna     | Madonna   | null     | null    |
# |  4|Jean-Claude Van...| Jean      |..Van     | Damme   |
# |  5|Dr. John Michael..| Dr.       |John...   | III     |
# +---+------------------+-----------+----------+---------+

spark.stop()
