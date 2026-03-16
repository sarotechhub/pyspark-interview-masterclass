"""
Q4: Select and WithColumn Operations — Transform columns effectively

Scenario: Learn the difference between select() and withColumn() operations
and how to transform columns using PySpark functions.

Key Concepts:
- select(): Choose/rename/transform specific columns, drops others
- withColumn(): Add/replace ONE column, keeps all others
- Alias: Rename columns in output
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - select(): O(n*m) — reads n rows, m selected columns
#   - withColumn(): O(n) — adds one column to n rows
#   - Multiple withColumn() chains: O(n*k) — recomputes all columns k times
#
# Shuffle Operations:
#   - select(): NO SHUFFLE (element-wise column selection)
#   - withColumn(): NO SHUFFLE (in-memory transformation)
#   - select() + filter(): NO SHUFFLE (both are narrow operations)
#
# Performance Tips:
#   - Use select() to reduce columns BEFORE joins (reduces data transferred)
#   - Chain select() operations instead of multiple withColumn()
#   - Avoid deep withColumn() nesting — use select() for complex transformations
#   - Push down column projection to CSV read for early filtering
#   - Use .drop() instead of complex select() for clarity
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Q04_Select_WithColumn") \
    .master("local[*]") \
    .getOrCreate()

print("=" * 80)
print("Q4: Select and WithColumn Operations")
print("=" * 80)

# ============================================================================
# Sample Data
# ============================================================================
data = [
    (1, "Alice", 50000, "Engineering"),
    (2, "Bob", 75000, "Sales"),
    (3, "Carol", 60000, "Engineering"),
    (4, "Dave", 95000, "Sales"),
    (5, "Eve", 55000, "HR"),
]

df = spark.createDataFrame(data, ["id", "name", "salary", "department"])

print("\nOriginal DataFrame:")
df.show()
print(f"Columns: {df.columns}")

# ============================================================================
# METHOD 1: select() — Choose specific columns
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 1: select() — Choose specific columns ---")
print("=" * 80)

print("\n1a. Select single column:")
result = df.select("name")
result.show()

print("\n1b. Select multiple columns:")
result = df.select("name", "salary", "department")
result.show()
print(f"Result columns: {result.columns}")

print("\n1c. Select using F.col():")
result = df.select(F.col("name"), F.col("salary"))
result.show()

print("\n1d. Select with alias (rename) — KEY DIFFERENCE vs withColumn:")
result = df.select(
    F.col("name").alias("employee_name"),
    F.col("salary").alias("annual_salary")
)
result.show()
print(f"New columns: {result.columns}")

# ============================================================================
# METHOD 2: withColumn() — Add or replace ONE column
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 2: withColumn() — Add/replace columns ---")
print("=" * 80)

print("\n2a. Add new column (keeps all original columns):")
result = df.withColumn("raise_amount", F.col("salary") * 0.1)
result.show()
print(f"Result has {len(result.columns)} columns")

print("\n2b. Replace existing column:")
result = df.withColumn("salary", F.col("salary") * 1.1)  # 10% raise
result.show()

print("\n2c. Multiple withColumn() calls (chained):")
result = df \
    .withColumn("salary_level", 
        F.when(F.col("salary") > 80000, "Senior")
         .when(F.col("salary") > 60000, "Mid")
         .otherwise("Junior")
    ) \
    .withColumn("bonus", F.col("salary") * 0.05)
result.show()

# ============================================================================
# METHOD 3: Key Difference — select() vs withColumn()
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 3: select() vs withColumn() Comparison ---")
print("=" * 80)

print("\nUsing select() — only 2 columns in result:")
result1 = df.select(F.col("name"), F.col("salary") * 2)
result1.show()
print(f"Columns: {result1.columns}, Count: {len(result1.columns)}")

print("\nUsing withColumn() — all columns + new column:")
result2 = df.withColumn("salary_x2", F.col("salary") * 2)
result2.show()
print(f"Columns: {result2.columns}, Count: {len(result2.columns)}")

print("\n✓ KEY DIFFERENCE:")
print("  select()    → Returns ONLY specified columns (projection)")
print("  withColumn() → Returns ALL original + new/modified column")

# ============================================================================
# METHOD 4: Transforming Columns with Functions
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 4: Transform columns with PySpark functions ---")
print("=" * 80)

# 4a. Math operations
print("\n4a. Math operations:")
result = df.select(
    "name",
    F.col("salary"),
    (F.col("salary") + 1000).alias("salary_plus_1k"),
    (F.col("salary") * 1.2).alias("salary_with_20pct"),
    F.round(F.col("salary") / 12, 2).alias("monthly_salary")
)
result.show()

# 4b. String operations
print("\n4b. String operations:")
result = df.select(
    F.upper(F.col("name")).alias("name_upper"),
    F.lower(F.col("name")).alias("name_lower"),
    F.length(F.col("name")).alias("name_length"),
    F.concat(F.col("name"), F.lit(" - "), F.col("department")).alias("name_dept")
)
result.show()

# 4c. Conditional transformation
print("\n4c. Conditional transformation (case/when):")
result = df.select(
    "name",
    "salary",
    F.when(F.col("salary") > 80000, "Senior")
     .when(F.col("salary") > 60000, "Mid")
     .otherwise("Junior")
     .alias("level")
)
result.show()

# 4d. Null handling
print("\n4d. Null handling:")
df_with_nulls = df.withColumn("bonus", 
    F.when(F.col("salary") > 60000, F.col("salary") * 0.1)
     .otherwise(F.lit(None))
)
result = df_with_nulls.select(
    "name",
    F.coalesce(F.col("bonus"), F.lit(0)).alias("bonus_filled")
)
result.show()

# ============================================================================
# METHOD 5: Derived Columns and Aggregations
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 5: Derived columns and aggregations ---")
print("=" * 80)

print("\n5a. Create salary bands:")
result = df.select(
    "name",
    "salary",
    F.round(F.col("salary") / 10000).alias("salary_band")
)
result.show()

print("\n5b. Percentile of salary (window function preview):")
from pyspark.sql import Window
window = Window.orderBy("salary")
result = df.select(
    "name",
    "salary",
    F.percent_rank().over(window).alias("percentile_rank")
)
result.show()

# ============================================================================
# METHOD 6: Renaming Columns
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 6: Renaming Columns ---")
print("=" * 80)

print("\n6a. Rename with select() + alias():")
result = df.select(
    F.col("id").alias("employee_id"),
    F.col("name").alias("employee_name"),
    F.col("salary").alias("annual_salary")
)
result.show()
print(f"New columns: {result.columns}")

print("\n6b. Rename with withColumnRenamed():")
result = df.withColumnRenamed("name", "employee_name") \
           .withColumnRenamed("salary", "annual_salary")
result.show()
print(f"New columns: {result.columns}")

# ============================================================================
# METHOD 7: Dropping Columns
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 7: Dropping Columns ---")
print("=" * 80)

print("\n7a. Drop single column:")
result = df.drop("department")
result.show()

print("\n7b. Drop multiple columns:")
result = df.drop("id", "department")
result.show()

# Equivalent with select (inverse)
result2 = df.select("name", "salary")
result2.show()

# ============================================================================
# METHOD 8: Complex Transformation Example
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 8: Complex Transformation Example ---")
print("=" * 80)

print("\nTransform DataFrame with multiple operations:")
result = df \
    .select(
        F.col("name").alias("employee"),
        F.col("salary"),
        F.col("department")
    ) \
    .withColumn("monthly_pay", F.round(F.col("salary") / 12, 2)) \
    .withColumn("annual_raise_10pct", F.round(F.col("salary") * 0.1, 2)) \
    .withColumn("salary_level",
        F.when(F.col("salary") >= 80000, "Level 4 - Senior")
         .when(F.col("salary") >= 60000, "Level 3 - Mid")
         .when(F.col("salary") >= 50000, "Level 2 - Junior")
         .otherwise("Level 1 - Entry")
    ) \
    .select(
        "employee",
        "salary",
        "monthly_pay",
        "salary_level"
    )

result.show()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY — Select and WithColumn")
print("=" * 80)
print("""
✓ select() — CHOOSE & TRANSFORM specific columns:
  df.select("col1", "col2")
  df.select(F.col("col1").alias("new_name"))
  df.select(F.col("col1") * 2)
  → Returns ONLY specified columns

✓ withColumn() — ADD OR REPLACE ONE column:
  df.withColumn("new_col", expression)
  df.withColumn("col", F.col("col") * 2)  # Replace
  → Returns ALL original columns + new/modified column

✓ KEY DIFFERENCES (select vs withColumn):
  
  | Operation | select() | withColumn() |
  |-----------|----------|--------------|
  | Purpose | Choose columns | Add/replace columns |
  | Columns returned | ONLY specified | ALL + new |
  | Use case | Projection | Enrich with computed fields |
  
✓ COMMON TRANSFORMATIONS:
  - Math: F.col("a") + 10, F.col("a") * 1.1, F.round(), F.abs()
  - String: F.upper(), F.lower(), F.concat(), F.length()
  - Conditional: F.when().when().otherwise()
  - Null handling: F.coalesce(), F.fillna()
  - Alias: F.col("a").alias("new_name")

✓ BEST PRACTICES:
  1. Use select() for projection (reduce columns)
  2. Use withColumn() to enrich (add computed fields)
  3. Chain multiple withColumn() calls for clarity
  4. Use alias() to rename in select()
  5. Use withColumnRenamed() for simple renames
""")

print("=" * 80)
