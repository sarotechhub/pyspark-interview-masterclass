"""
Q1: Remove duplicates based on specific columns and keep the latest record.

Scenario: You have a customer table with duplicate customer_id rows. 
Keep only the most recent record per customer.
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - Window partitionBy: O(n) — single pass
#   - Window with orderBy: O(n*log(n)) — sorting per partition
#   - row_number() over window: O(n*log(n)) — ranking sorted data
#   - Filter rank == 1: O(n) — single pass
#   - Total: O(n*log(n)) — dominated by sorting
#
# Shuffle Operations:
#   - partitionBy(customer_id): FULL SHUFFLE (redistribute by key)
#   - orderBy(date): NO SHUFFLE (sort within partition)
#
# Performance Tips:
#   - This approach requires ONE shuffle operation
#   - Use dense_rank() if handling ties (no gaps)
#   - Filter on indexed column if customer_id has index
#   - Consider: dropDuplicates() for simpler dedup (no ordering)
#   - For very large data: repartition by customer_id first
#   - Keep only essential columns in window spec
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Q01_RemoveDuplicates") \
    .master("local[*]") \
    .getOrCreate()

# ============================================
# SAMPLE DATA
# ============================================
data = [
    (1, "Alice", "2024-01-15"),
    (1, "Alice", "2024-03-20"),  # Latest — keep this
    (2, "Bob",   "2024-02-10"),
    (2, "Bob",   "2024-01-05"),  # Older — drop this
    (3, "Carol", "2024-02-28"),
]

# Create DataFrame with schema
df = spark.createDataFrame(
    data,
    ["customer_id", "name", "created_date"]
)

print("=" * 60)
print("INPUT DATA:")
print("=" * 60)
df.show()

# ============================================
# SOLUTION
# ============================================
# Step 1: Create a window that partitions by customer_id
#         and orders by created_date in descending order
window = Window.partitionBy("customer_id").orderBy(F.desc("created_date"))

# Step 2: Add a row number column. row_number() = 1 for the most recent record
#         per customer (since we ordered DESC)
result = df \
    .withColumn("rank", F.row_number().over(window)) \
    .filter(F.col("rank") == 1) \
    .drop("rank")

print("\n" + "=" * 60)
print("OUTPUT DATA (Latest record per customer):")
print("=" * 60)
result.show()

# ============================================
# EXPECTED OUTPUT:
# ============================================
# +----------+-----+------------+
# |customer_id|name|created_date|
# +----------+-----+------------+
# |         1|Alice| 2024-03-20|
# |         2|  Bob| 2024-02-10|
# |         3|Carol| 2024-02-28|
# +----------+-----+------------+

# Optional: Alternative with dense_rank() for ties
# dense_rank() will keep all records with the same latest date
result_dense = df \
    .withColumn("rank", F.dense_rank().over(window)) \
    .filter(F.col("rank") == 1) \
    .drop("rank")

print("\n" + "=" * 60)
print("OUTPUT DATA (Using dense_rank for ties):")
print("=" * 60)
result_dense.show()

spark.stop()
