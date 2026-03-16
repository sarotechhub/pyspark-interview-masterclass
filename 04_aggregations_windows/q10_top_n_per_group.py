"""
Q10: Find the top N products per category by revenue.

Scenario: Rank and filter to show top products in each category.
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - partitionBy(category): O(n) — redistribute
#   - orderBy(revenue DESC): O(n*log(n)) — sort per partition
#   - dense_rank(): O(n) — ranking within sorted partition
#   - filter(rank <= N): O(n) — final filter
#   - Total: O(n*log(n)) — dominated by sorting
#
# Shuffle Operations:
#   - partitionBy(): SHUFFLE (redistribute by category)
#   - orderBy(): NO SHUFFLE (sort within partition)
#   - One shuffle operation total
#
# Performance Tips:
#   - Use dense_rank() for ties, row_number() for arbitrary breaks
#   - N should be small (top 3, 5, 10) to limit output
#   - Filter applied AFTER ranking (correct approach)
#   - Cache before multiple top-N queries on same data
#   - Low-cardinality groupBy columns preferred
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Q10_TopNPerCategory") \
    .master("local[*]") \
    .getOrCreate()

# ============================================
# SAMPLE DATA
# ============================================
data = [
    ("Electronics", "TV",     5000),
    ("Electronics", "Phone",  3000),
    ("Electronics", "Laptop", 7000),
    ("Electronics", "Tablet", 2000),
    ("Clothing",    "Jacket", 1500),
    ("Clothing",    "Shoes",  2500),
    ("Clothing",    "Shirt",   500),
    ("Clothing",    "Pants",  1800),
    ("Home",        "Sofa",   3500),
    ("Home",        "Chair",  1200),
    ("Home",        "Table",  2000),
]

df = spark.createDataFrame(data, ["category", "product", "revenue"])

print("=" * 60)
print("INPUT DATA:")
print("=" * 60)
df.show()

# ============================================
# SOLUTION
# ============================================

# Parameter: Top N products per category
N = 2

# Step 1: Create a window ordered by revenue (descending) within each category
window = Window.partitionBy("category").orderBy(F.desc("revenue"))

# Step 2: Add a rank column using dense_rank()
#         dense_rank(): no gaps in ranking when there are ties
#         row_number(): always sequential (good when no ties expected)
#         rank(): gaps when ties exist
result = df \
    .withColumn("rank", F.dense_rank().over(window)) \
    .filter(F.col("rank") <= N) \
    .drop("rank")

print("\n" + "=" * 60)
print("OUTPUT DATA (Top 2 products per category):")
print("=" * 60)
result.show()

# ============================================
# ALTERNATIVE 1: Using row_number() instead
# ============================================
# row_number() assigns a unique sequential number to each row
# If there are ties in revenue, it will break the tie arbitrarily
result_row_number = df \
    .withColumn("rank", F.row_number().over(window)) \
    .filter(F.col("rank") <= N) \
    .drop("rank")

print("\n" + "=" * 60)
print("ALTERNATIVE (Using row_number):")
print("=" * 60)
result_row_number.show()

# ============================================
# ALTERNATIVE 2: Include ranking details
# ============================================
# Show the rank and revenue for clarity
result_with_rank = df \
    .withColumn("rank", F.dense_rank().over(window)) \
    .withColumn("revenue_rank_pct",
        F.round(F.percent_rank().over(window) * 100, 2)
    ) \
    .filter(F.col("rank") <= N)

print("\n" + "=" * 60)
print("WITH RANKING DETAILS:")
print("=" * 60)
result_with_rank.show()

# ============================================
# ALTERNATIVE 3: Top N with ties
# ============================================
# If there's a tie for the Nth position, include all tied products
# This might return more than N products per category

N_cutoff = df \
    .withColumn("rank", F.dense_rank().over(window)) \
    .filter(F.col("rank") == N) \
    .agg(F.max("revenue")).collect()[0][0]

# Find all products with revenue >= the Nth rank's revenue
result_with_ties = df \
    .withColumn("rank", F.dense_rank().over(window)) \
    .filter(
        (F.col("rank") < N) |
        (F.col("rank") == N)
    )

print("\n" + "=" * 60)
print("INCLUDING TIES (May have more than N due to ties):")
print("=" * 60)
result_with_ties.show()

# ============================================
# EXPECTED OUTPUT (Top 2):
# ============================================
# +-----------+-------+-------+
# |   category|product|revenue|
# +-----------+-------+-------+
# |Electronics| Laptop|   7000|
# |Electronics|    TV |   5000|
# |   Clothing|  Shoes|   2500|
# |   Clothing| Jacket|   1500|
# |       Home|  Sofa |   3500|
# |       Home|  Table|   2000|
# +-----------+-------+-------+

spark.stop()
