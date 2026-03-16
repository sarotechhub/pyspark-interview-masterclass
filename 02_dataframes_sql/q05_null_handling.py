"""
Q5: Null Handling — Deal with missing values effectively

Scenario: Learn various strategies to handle null values in DataFrames.

Key Concepts:
- dropna(): Remove rows with nulls
- fillna(): Fill nulls with values
- coalesce(): Get first non-null value
- isNull() / isNotNull(): Check for nulls
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - dropna(): O(n) — scans each row, filters some
#   - fillna(): O(n) — updates null cells
#   - coalesce(): O(n*k) — checks k columns per row
#   - filter(isNull()): O(n) — scans each row
#
# Shuffle Operations:
#   - dropna(): NO SHUFFLE (row filtering)
#   - fillna(): NO SHUFFLE (column update)
#   - Any null check: NO SHUFFLE (narrow operation)
#
# Performance Tips:
#   - Use dropna(how='any', subset=[...]) to limit scope
#   - fillna(value) is faster than fillna(column_map)
#   - Use coalesce() to handle multiple fallback columns efficiently
#   - Filter nulls EARLY in pipeline to reduce data
#   - Consider null handling at load time (CSV options)
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Q05_Null_Handling") \
    .master("local[*]") \
    .getOrCreate()

print("=" * 80)
print("Q5: Null Handling in PySpark")
print("=" * 80)

# ============================================================================
# Sample Data with Nulls
# ============================================================================
data = [
    (1, "Alice", 50000, "Engineering"),
    (2, None, 75000, "Sales"),            # Null name
    (3, "Carol", None, "Engineering"),     # Null salary
    (4, "Dave", 95000, None),              # Null department
    (5, "Eve", 55000, "HR"),
    (6, None, None, "Finance"),            # Multiple nulls
]

df = spark.createDataFrame(data, ["id", "name", "salary", "department"])

print("\nOriginal DataFrame with nulls:")
df.show()

# ============================================================================
# METHOD 1: dropna() — Remove rows with nulls
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 1: dropna() — Remove rows with nulls ---")
print("=" * 80)

print("\n1a. Drop rows with ANY null:")
result = df.dropna()
result.show()
print(f"Rows: {result.count()} (was {df.count()})")

print("\n1b. Drop rows where specific columns are null:")
result = df.dropna(subset=["name", "salary"])
result.show()
print(f"Rows: {result.count()}")

print("\n1c. Drop rows where at least N non-null values:")
result = df.dropna(thresh=3)  # Keep rows with at least 3 non-null values
result.show()

print("\n1d. Drop rows if ANY of specific columns are null:")
result = df.dropna(how="any", subset=["name", "salary"])
result.show()

print("\n1e. Drop rows if ALL of specific columns are null:")
result = df.dropna(how="all", subset=["name", "salary"])
result.show()

# ============================================================================
# METHOD 2: fillna() — Replace nulls with values
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 2: fillna() — Replace nulls ---")
print("=" * 80)

print("\n2a. Fill all nulls with single value:")
result = df.fillna("Unknown")
result.show()

print("\n2b. Fill nulls in specific column:")
result = df.fillna(value=0, subset=["salary"])
result.show()

print("\n2c. Fill different columns with different values:")
result = df.fillna({
    "name": "Unknown",
    "salary": 0,
    "department": "Unassigned"
})
result.show()

print("\n2d. Fill numeric columns with mean:")
mean_salary = df.selectExpr("percentile_approx(salary, 0.5)").collect()[0][0]
print(f"Mean salary: {mean_salary}")
result = df.fillna({"salary": mean_salary})
result.show()

# ============================================================================
# METHOD 3: coalesce() — Get first non-null value
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 3: coalesce() — First non-null value ---")
print("=" * 80)

# Create data with alternative columns
data2 = [
    (1, "Alice", 50000, None, 52000),       # salary=50000
    (2, None, 75000, "Bob Smith", 76000),   # name=None, use alt_name
    (3, "Carol", None, None, 62000),        # salary=None, use alt_salary
]
df2 = spark.createDataFrame(data2, ["id", "name", "salary", "alt_name", "alt_salary"])

print("\nOriginal DataFrame:")
df2.show()

print("\n3a. Use coalesce() for name (name OR alt_name):")
result = df2.select(
    "id",
    F.coalesce(F.col("name"), F.col("alt_name")).alias("name"),
    F.col("salary")
)
result.show()

print("\n3b. Use coalesce() for salary (salary OR alt_salary):")
result = df2.select(
    "id",
    "name",
    F.coalesce(F.col("salary"), F.col("alt_salary")).alias("salary")
)
result.show()

print("\n3c. Coalesce multiple columns:")
result = df2.select(
    "id",
    F.coalesce(F.col("name"), F.col("alt_name"), F.lit("Unknown")).alias("name"),
    F.coalesce(F.col("salary"), F.col("alt_salary"), F.lit(0)).alias("salary")
)
result.show()

# ============================================================================
# METHOD 4: Conditional filling with when/otherwise
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 4: Conditional filling with when/otherwise ---")
print("=" * 80)

result = df.select(
    "id",
    "name",
    "salary",
    "department",
    F.when(F.col("salary").isNull(), 0)
     .otherwise(F.col("salary"))
     .alias("salary_filled"),
    F.when(F.col("department").isNull(), "Unassigned")
     .otherwise(F.col("department"))
     .alias("department_filled")
)

print("\nFill with conditional logic:")
result.show()

# ============================================================================
# METHOD 5: Check for nulls — isNull() / isNotNull()
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 5: Check for nulls ---")
print("=" * 80)

print("\n5a. Filter for null values:")
result = df.filter(F.col("name").isNull())
result.show()
print(f"Rows with null name: {result.count()}")

print("\n5b. Filter for non-null values:")
result = df.filter(F.col("salary").isNotNull())
result.show()
print(f"Rows with non-null salary: {result.count()}")

print("\n5c. Multiple conditions:")
result = df.filter(
    (F.col("name").isNotNull()) & 
    (F.col("salary").isNotNull())
)
result.show()

print("\n5d. Add flag for nullness:")
result = df.select(
    "id",
    "name",
    "salary",
    F.col("salary").isNull().alias("salary_is_null"),
    F.col("department").isNull().alias("dept_is_null")
)
result.show()

# ============================================================================
# METHOD 6: nanvl() — Replace NaN with value (for numeric)
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 6: Replace NaN values ---")
print("=" * 80)

data3 = [
    (1, 10.0),
    (2, float('nan')),
    (3, 30.0),
]
df3 = spark.createDataFrame(data3, ["id", "value"])

print("\nDataFrame with NaN:")
df3.show()

print("\nFill NaN with value:")
result = df3.select(
    "id",
    F.when(F.isnan("value"), 0)
     .otherwise(F.col("value"))
     .alias("value_filled")
)
result.show()

# ============================================================================
# METHOD 7: Data quality checks
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 7: Data quality analysis ---")
print("=" * 80)

print("\nNull count per column:")
null_counts = df.select([
    F.count(F.when(F.col(c).isNull(), 1)).alias(c) 
    for c in df.columns
])
null_counts.show()

print("\nNull percentage per column:")
total_rows = df.count()
for col in df.columns:
    null_count = df.filter(F.col(col).isNull()).count()
    null_pct = (null_count / total_rows) * 100
    print(f"  {col}: {null_count}/{total_rows} ({null_pct:.1f}%)")

# ============================================================================
# METHOD 8: Complex null handling strategy
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 8: Complete null handling strategy ---")
print("=" * 80)

print("\nApply comprehensive null handling:")
result = df \
    .fillna({        # First pass: fill with defaults
        "name": "Unknown",
        "salary": 0,
        "department": "Unassigned"
    }) \
    .filter(          # Remove rows where critical cols are still null
        F.col("id").isNotNull()
    ) \
    .select(          # Add quality flags
        "*",
        F.when(F.col("salary") == 0, True)
         .otherwise(False)
         .alias("salary_was_null")
    )

result.show()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY — Null Handling")
print("=" * 80)
print("""
✓ dropna() — REMOVE ROWS WITH NULLS:
  df.dropna()                          # Drop any row with null
  df.dropna(subset=["col1", "col2"])  # Check specific columns
  df.dropna(thresh=2)                 # Keep if 2+ non-null values
  df.dropna(how="all")                # Drop if ALL are null

✓ fillna() — FILL NULLS WITH VALUES:
  df.fillna(0)                        # Fill all with value
  df.fillna({"col1": 0, "col2": ""}) # Fill different columns
  df.fillna(mean_value)               # Fill with computed value

✓ coalesce() — FIRST NON-NULL VALUE:
  F.coalesce(col1, col2, col3)        # Return first non-null
  F.coalesce(col, F.lit("default"))   # Fallback to literal

✓ Check for nulls:
  F.col("name").isNull()              # Is this null?
  F.col("name").isNotNull()           # Is this not null?
  F.isnan("value")                    # Is this NaN?

✓ BEST PRACTICES:
  1. Analyze null distribution FIRST
  2. Decide: drop or fill? Why?
  3. Use appropriate strategy:
     - DROP: If nulls are errors/rare
     - FILL: If nulls represent missing data
     - COALESCE: If alternative data exists
  4. Document null imputation rationale
  5. Add flags for filled values (transparency)
  6. Validate after null handling
""")

print("=" * 80)
