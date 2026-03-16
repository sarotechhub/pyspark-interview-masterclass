"""
Q3: Pivot a table — convert row values into columns.

Scenario: Sales data has (year, product, amount) rows. 
Pivot to show products as columns.
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - groupBy() part: O(n) — aggregate per group
#   - pivot() operation: O(n*k) — k = number of pivot values
#   - agg() functions: O(k) — one per pivot value
#   - Total: O(n*k) where k = distinct pivot values
#
# Shuffle Operations:
#   - Full pivot: FULL SHUFFLE (redistributes by groupBy columns)
#   - agg() after groupBy: SHUFFLE operations
#   - Pre-filtering: NO SHUFFLE (before groupBy)
#
# Performance Tips:
#   - SPECIFY pivot values (avoids scanning for unique values)
#   - Pre-filter rows before pivot to reduce data
#   - Pivot on low-cardinality columns (few distinct values)
#   - Avoid pivot on high-cardinality columns (100s of values)
#   - Multiple small pivots better than one large pivot
#   - Consider spark.sql.pivotMaxValues config (default 10000)
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Q03_PivotTable") \
    .master("local[*]") \
    .getOrCreate()

# ============================================
# SAMPLE DATA
# ============================================
data = [
    (2023, "TV",     5000),
    (2023, "Phone",  3000),
    (2023, "Laptop", 7000),
    (2024, "TV",     5500),
    (2024, "Phone",  3200),
    (2024, "Laptop", 8000),
]

df = spark.createDataFrame(data, ["year", "product", "amount"])

print("=" * 60)
print("INPUT DATA (Long Format):")
print("=" * 60)
df.show()

# ============================================
# SOLUTION
# ============================================
# Step 1: Use groupBy() on the row dimension (year)
# Step 2: Use pivot() to specify which column becomes new columns (product)
# Step 3: Pass the list of expected pivot values for better performance
#         (Optional but recommended to avoid scanning for unique values)
# Step 4: Use agg() to aggregate values (sum, count, mean, etc.)

result = df.groupBy("year") \
    .pivot("product", ["TV", "Phone", "Laptop"]) \
    .agg(F.sum("amount"))

print("\n" + "=" * 60)
print("OUTPUT DATA (Wide Format - Pivoted):")
print("=" * 60)
result.show()

# ============================================
# Alternative: Without specifying pivot values
# ============================================
# This scans the DataFrame to find unique product values
# Less efficient for large datasets but more robust
result_dynamic = df.groupBy("year") \
    .pivot("product") \
    .agg(F.sum("amount"))

print("\n" + "=" * 60)
print("OUTPUT DATA (Dynamic Pivot - No explicit values):")
print("=" * 60)
result_dynamic.show()

# ============================================
# Alternative: Multiple aggregations
# ============================================
result_multi_agg = df.groupBy("year") \
    .pivot("product", ["TV", "Phone", "Laptop"]) \
    .agg(
        F.sum("amount").alias("total_sales"),
        F.count("*").alias("transaction_count")
    )

print("\n" + "=" * 60)
print("OUTPUT DATA (Multiple Aggregations):")
print("=" * 60)
result_multi_agg.show()

# ============================================
# EXPECTED OUTPUT (First method):
# ============================================
# +----+----+-----+------+
# |year|  TV|Phone|Laptop|
# +----+----+-----+------+
# |2023|5000| 3000|  7000|
# |2024|5500| 3200|  8000|
# +----+----+-----+------+

# ============================================
# EXPECTED OUTPUT (Multiple aggregations):
# ============================================
# +----+------------------+-----+--+-----+--+
# |year|TV_total_sales|...|Phone_...|Laptop_...|
# +----+------------------+-----+--+-----+--+

spark.stop()
