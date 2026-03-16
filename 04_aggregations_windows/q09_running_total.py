"""
Q9: Calculate running total (cumulative sum) per customer.

Scenario: For each customer, calculate the cumulative sum of their orders.
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - partitionBy(customer_id): O(n) — redistribute
#   - orderBy(order_date): O(n*log(n)) — sort per partition
#   - sum() over window: O(n*k) — k = rows in window (unbounded)
#   - Total: O(n*log(n) + n*k) = O(n*k) for large windows
#
# Shuffle Operations:
#   - partitionBy(): SHUFFLE (redistribute by customer)
#   - orderBy(): NO SHUFFLE (sort within partition)
#   - One shuffle operation total
#
# Performance Tips:
#   - unboundedPreceding to currentRow is memory efficient
#   - Data must fit in executor memory for large partitions
#   - Large customer partitions may cause bottleneck
#   - Consider repartition(customer_id) before window
#   - Cache result if used multiple times
#   - Use specific columns in window spec (not *)
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Q09_RunningTotal") \
    .master("local[*]") \
    .getOrCreate()

# ============================================
# SAMPLE DATA
# ============================================
data = [
    (1, "2024-01-01", 100),
    (1, "2024-01-05", 200),
    (1, "2024-01-10", 150),
    (2, "2024-01-02", 300),
    (2, "2024-01-08", 100),
    (2, "2024-01-15", 250),
]

df = spark.createDataFrame(data, ["customer_id", "order_date", "amount"])

print("=" * 60)
print("INPUT DATA:")
print("=" * 60)
df.show()

# ============================================
# SOLUTION
# ============================================

# Step 1: Define a window that:
#         - Partitions by customer_id (separate totals per customer)
#         - Orders by order_date (to accumulate in time order)
#         - rowsBetween(unbounded, current) means: from first row to current row
window = Window.partitionBy("customer_id") \
               .orderBy("order_date") \
               .rowsBetween(Window.unboundedPreceding, Window.currentRow)

# Step 2: Calculate the running sum using the window
#         sum() aggregates all amounts within the window boundaries
result = df.withColumn("running_total", F.sum("amount").over(window))

print("\n" + "=" * 60)
print("OUTPUT DATA (Running Total per Customer):")
print("=" * 60)
result.show()

# ============================================
# ALTERNATIVE: Using rangeBetween for date-based windows
# ============================================
# If you want running total over a date range (e.g., last 7 days)

SECONDS_PER_DAY = 86400

# Convert order_date to numeric (seconds since epoch)
df_range = df.withColumn("date_long", F.col("order_date").cast("long"))

# Window with rangeBetween: 7 days before current date to current date
window_7day = Window.partitionBy("customer_id") \
                    .orderBy("date_long") \
                    .rangeBetween(-7 * SECONDS_PER_DAY, 0)

result_7day = df_range.withColumn(
    "running_total_7days",
    F.sum("amount").over(window_7day)
)

print("\n" + "=" * 60)
print("ALTERNATIVE: Running Total (Last 7 days per date):")
print("=" * 60)
result_7day.select("customer_id", "order_date", "amount", "running_total_7days").show()

# ============================================
# ADVANCED: Include other window functions
# ============================================

window_adv = Window.partitionBy("customer_id") \
                   .orderBy("order_date") \
                   .rowsBetween(Window.unboundedPreceding, Window.currentRow)

result_advanced = df \
    .withColumn("running_total", F.sum("amount").over(window_adv)) \
    .withColumn("running_count", F.count("*").over(window_adv)) \
    .withColumn("running_avg", 
        F.round(F.avg("amount").over(window_adv), 2)
    ) \
    .withColumn("max_so_far", F.max("amount").over(window_adv)) \
    .withColumn("min_so_far", F.min("amount").over(window_adv))

print("\n" + "=" * 60)
print("ADVANCED: Multiple Running Aggregates:")
print("=" * 60)
result_advanced.show()

# ============================================
# EXPECTED OUTPUT:
# ============================================
# +-----------+----------+------+-----------+
# |customer_id|order_date|amount|running_tot|
# +-----------+----------+------+-----------+
# |          1|2024-01-01|   100|        100|
# |          1|2024-01-05|   200|        300|
# |          1|2024-01-10|   150|        450|
# |          2|2024-01-02|   300|        300|
# |          2|2024-01-08|   100|        400|
# |          2|2024-01-15|   250|        650|
# +-----------+----------+------+-----------+

spark.stop()
