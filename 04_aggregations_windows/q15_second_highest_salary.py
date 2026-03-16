"""
Q15: Find the second highest salary in each department.

Scenario: Identify the runner-up earner in each department.
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - partitionBy(dept): O(n) — redistribute
#   - orderBy(salary DESC): O(n*log(n)) — sort per partition
#   - dense_rank(): O(n) — ranking (no gaps for ties)
#   - filter(rank == 2): O(n) — final filter
#   - row_number(): O(n) — ranking (breaks ties arbitrarily)
#   - Total: O(n*log(n)) — dominated by sorting
#
# Shuffle Operations:
#   - partitionBy(): SHUFFLE (redistribute by department)
#   - orderBy(): NO SHUFFLE (sort within partition)
#   - One shuffle operation total
#
# Performance Tips:
#   - dense_rank() keeps tied salaries (same rank)
#   - row_number() breaks ties arbitrarily (sequential rank)
#   - Choose rank function based on tie handling needed
#   - Filter rank==2 is efficient (happens after ranking)
#   - Low-cardinality groupBy (departments) preferred
#   - Cache if finding multiple percentiles (2nd, 3rd, etc.)
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Q15_SecondHighestSalary") \
    .master("local[*]") \
    .getOrCreate()

# ============================================
# SAMPLE DATA
# ============================================
data = [
    ("Engineering", "Alice", 120000),
    ("Engineering", "Bob",   95000),
    ("Engineering", "Carol", 110000),
    ("Engineering", "David", 95000),   # Tie with Bob
    ("Marketing",   "Eve",   85000),
    ("Marketing",   "Frank", 90000),
    ("Marketing",   "Grace", 80000),
    ("Sales",       "Henry", 75000),
    ("Sales",       "Ivy",   80000),
]

df = spark.createDataFrame(data, ["dept", "employee", "salary"])

print("=" * 60)
print("INPUT DATA:")
print("=" * 60)
df.show()

# ============================================
# SOLUTION - METHOD 1: Using dense_rank()
# ============================================
# dense_rank() gives same rank to ties, no gaps

window = Window.partitionBy("dept").orderBy(F.desc("salary"))

result_dense = df \
    .withColumn("dense_rank", F.dense_rank().over(window)) \
    .filter(F.col("dense_rank") == 2) \
    .select("dept", "employee", "salary")

print("\n" + "=" * 60)
print("METHOD 1: Using dense_rank() (handles ties):")
print("=" * 60)
result_dense.show()

# ============================================
# SOLUTION - METHOD 2: Using row_number()
# ============================================
# row_number() breaks ties arbitrarily

result_row = df \
    .withColumn("row_num", F.row_number().over(window)) \
    .filter(F.col("row_num") == 2) \
    .select("dept", "employee", "salary")

print("\n" + "=" * 60)
print("METHOD 2: Using row_number() (breaks ties):")
print("=" * 60)
result_row.show()

# ============================================
# SOLUTION - METHOD 3: Using rank()
# ============================================
# rank() gives same rank to ties, gaps appear

result_rank = df \
    .withColumn("rank", F.rank().over(window)) \
    .filter(F.col("rank") == 2) \
    .select("dept", "employee", "salary")

print("\n" + "=" * 60)
print("METHOD 3: Using rank() (has gaps with ties):")
print("=" * 60)
result_rank.show()

# ============================================
# ADVANCED: Show all ranking methods comparison
# ============================================

comparison = df \
    .withColumn("row_num", F.row_number().over(window)) \
    .withColumn("rank", F.rank().over(window)) \
    .withColumn("dense_rank", F.dense_rank().over(window)) \
    .withColumn("percent_rank", F.round(F.percent_rank().over(window), 2))

print("\n" + "=" * 60)
print("COMPARISON: All Ranking Methods:")
print("=" * 60)
comparison.show()

# ============================================
# ADVANCED: Top 3 per department with salary info
# ============================================

top_3 = df \
    .withColumn("rank", F.dense_rank().over(window)) \
    .filter(F.col("rank") <= 3) \
    .withColumn("rank_name",
        F.case_when(F.col("rank") == 1, "Highest")
         .when(F.col("rank") == 2, "Second Highest")
         .when(F.col("rank") == 3, "Third Highest")
    )

print("\n" + "=" * 60)
print("TOP 3 PER DEPARTMENT (with labels):")
print("=" * 60)
top_3.select("dept", "rank_name", "employee", "salary").show()

# ============================================
# ADVANCED: Statistics per department
# ============================================

dept_stats = df.groupBy("dept").agg(
    F.max("salary").alias("highest_salary"),
    F.count("*").alias("emp_count"),
    F.avg("salary").alias("avg_salary")
)

# Join with second highest to show context
second_highest = df \
    .withColumn("dense_rank", F.dense_rank().over(window)) \
    .filter(F.col("dense_rank") == 2) \
    .select(F.col("dept"), F.col("salary").alias("second_highest_salary"))

dept_context = dept_stats.join(second_highest, on="dept", how="left")

print("\n" + "=" * 60)
print("DEPARTMENT SALARY CONTEXT:")
print("=" * 60)
dept_context.show()

# ============================================
# EXPECTED OUTPUT (dense_rank):
# ============================================
# +------------+--------+------+
# |         dept|employee|salary|
# +------------+--------+------+
# |Engineering | Carol  |110000|
# |  Marketing | Frank  | 90000|
# |     Sales  | Ivy    | 80000|
# +------------+--------+------+

# ============================================
# EXPLANATION:
# ============================================
# - Engineering: Alice (120k) is highest, Carol (110k) is second
# - Marketing: Frank (90k) is second (Eve has 85k)
# - Sales: Ivy (80k) is second (Henry has 75k)
#
# With ties (Bob & David both 95k):
# - dense_rank would show only 1 second highest (Carol at 110k)
# - rank would skip to position 3 if there were ties at position 2
# - row_number would pick one arbitrarily

spark.stop()
