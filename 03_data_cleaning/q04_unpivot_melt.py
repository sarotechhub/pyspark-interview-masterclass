"""
Q4: Unpivot (melt) columns back into rows.

Scenario: You have a wide table with product columns. 
Convert back to long format.
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - stack() method: O(n*k) — expands n rows to n*k rows
#   - union() of DataFrames: O(n*k) — combines all rows
#   - select() on unpivot: O(n*k) — reads expanded data
#   - Total: O(n*k) where k = number of columns unpivoted
#
# Shuffle Operations:
#   - stack() unpivot: NO SHUFFLE (transforms rows)
#   - union(): NO SHUFFLE (just concatenates)
#   - No shuffles in basic unpivot operation
#
# Performance Tips:
#   - Data expands n rows × k columns → n*k rows
#   - Plan downstream operations for increased data size
#   - stack() faster than multiple union() calls
#   - Keep unpivot logic together (avoid intermediate outputs)
#   - Filter null values AFTER unpivot for clarity
#   - Consider partition count increase after unpivot
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Q04_UnpivotMelt") \
    .master("local[*]") \
    .getOrCreate()

# ============================================
# SAMPLE DATA
# ============================================
# Wide format: each product is a column
data = [(2023, 5000, 3000, 7000),
        (2024, 5500, 3200, 8000)]

df = spark.createDataFrame(data, ["year", "TV", "Phone", "Laptop"])

print("=" * 60)
print("INPUT DATA (Wide Format):")
print("=" * 60)
df.show()

# ============================================
# SOLUTION - METHOD 1: Using stack()
# ============================================
# stack(n, col1, val1, col2, val2, ...) creates n rows per input row
# Format: stack(number_of_rows, name1, value1, name2, value2, ...)

result_method1 = df.select(
    "year",
    # stack(3, ...) creates 3 rows per input row
    # 'TV' becomes the product name, TV becomes the value
    # 'Phone' becomes the product name, Phone becomes the value
    # 'Laptop' becomes the product name, Laptop becomes the value
    F.expr("stack(3, 'TV', TV, 'Phone', Phone, 'Laptop', Laptop) as (product, amount)")
)

print("\n" + "=" * 60)
print("OUTPUT METHOD 1 (Using stack()):")
print("=" * 60)
result_method1.show()

# ============================================
# SOLUTION - METHOD 2: Using unionByName with select
# ============================================
# Create separate dataframes for each column, then union them

tv_df = df.select("year", F.lit("TV").alias("product"), F.col("TV").alias("amount"))
phone_df = df.select("year", F.lit("Phone").alias("product"), F.col("Phone").alias("amount"))
laptop_df = df.select("year", F.lit("Laptop").alias("product"), F.col("Laptop").alias("amount"))

# Union all dataframes
result_method2 = tv_df.unionByName(phone_df).unionByName(laptop_df)

print("\n" + "=" * 60)
print("OUTPUT METHOD 2 (Using unionByName):")
print("=" * 60)
result_method2.show()

# ============================================
# SOLUTION - METHOD 3: Using melt (Spark 3.4+)
# ============================================
# Note: melt() function is available in Spark 3.4+
# It's more intuitive than stack()

# Uncomment if using Spark 3.4+
# result_method3 = df.melt(
#     ids=["year"],
#     values=["TV", "Phone", "Laptop"],
#     variableColumnName="product",
#     valueColumnName="amount"
# )

# ============================================
# EXPECTED OUTPUT:
# ============================================
# +----+-------+------+
# |year|product|amount|
# +----+-------+------+
# |2023|     TV|  5000|
# |2023|  Phone|  3000|
# |2023| Laptop|  7000|
# |2024|     TV|  5500|
# |2024|  Phone|  3200|
# |2024| Laptop|  8000|
# +----+-------+------+

spark.stop()
