"""
Q13: Calculate 7-day rolling average of sales.

Scenario: Smooth daily sales data with a 7-day moving average.
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - partitionBy(store_id): O(n) — redistribute
#   - orderBy(sale_date): O(n*log(n)) — sort per partition
#   - rowsBetween(-6, 0): O(n*7) = O(n) — fixed 7-row window
#   - average calculation: O(n) — aggregation
#   - Total: O(n*log(n)) — dominated by sorting
#
# Shuffle Operations:
#   - partitionBy(): SHUFFLE (redistribute by store)
#   - No shuffle for window ordering
#
# Performance Tips:
#   - Fixed window size (rowsBetween) is memory efficient
#   - First 6 rows have partial windows (not full 7)
#   - Unbounded window (range between) less efficient
#   - Filter nulls after rolling avg (first 6 rows may be partial)
#   - Cache if computing multiple rolling windows
#   - Pre-filter date range to reduce data
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Q13_RollingAverage") \
    .master("local[*]") \
    .getOrCreate()

# ============================================
# SAMPLE DATA
# ============================================
data = [
    ("store_1", "2024-01-01", 1000),
    ("store_1", "2024-01-02", 1200),
    ("store_1", "2024-01-03", 900),
    ("store_1", "2024-01-04", 1100),
    ("store_1", "2024-01-05", 950),
    ("store_1", "2024-01-06", 1300),
    ("store_1", "2024-01-07", 1150),
    ("store_1", "2024-01-08", 1250),
    ("store_1", "2024-01-09", 1100),
    ("store_2", "2024-01-01", 800),
    ("store_2", "2024-01-02", 850),
    ("store_2", "2024-01-03", 920),
    ("store_2", "2024-01-04", 880),
    ("store_2", "2024-01-05", 950),
    ("store_2", "2024-01-06", 1000),
    ("store_2", "2024-01-07", 920),
]

df = spark.createDataFrame(data, ["store_id", "sale_date", "sales"])

# Convert sale_date to actual date type and create numeric version
df = df.withColumn("sale_date", F.to_date("sale_date")) \
       .withColumn("date_long", F.col("sale_date").cast("long"))

print("=" * 60)
print("INPUT DATA:")
print("=" * 60)
df.show()

# ============================================
# SOLUTION - METHOD 1: Using rowsBetween
# ============================================

# Define the window: 7 days of history (including current day)
# rowsBetween(-6, 0) means: 6 rows back + current row = 7 rows total
window_rows = Window.partitionBy("store_id") \
                    .orderBy("sale_date") \
                    .rowsBetween(-6, 0)  # 7 rows: 6 back + current

result_rows = df.withColumn("rolling_7d_avg_rows",
    F.round(F.avg("sales").over(window_rows), 2)
)

print("\n" + "=" * 60)
print("METHOD 1: 7-Day Rolling Average (rowsBetween):")
print("=" * 60)
result_rows.select("store_id", "sale_date", "sales", "rolling_7d_avg_rows").show()

# ============================================
# SOLUTION - METHOD 2: Using rangeBetween with dates
# ============================================

SECONDS_PER_DAY = 86400

# rangeBetween works with numeric values
# -6 * SECONDS_PER_DAY = 6 days back (in seconds)
# 0 = current day
window_range = Window.partitionBy("store_id") \
                     .orderBy("date_long") \
                     .rangeBetween(-6 * SECONDS_PER_DAY, 0)

result_range = df.withColumn("rolling_7d_avg_range",
    F.round(F.avg("sales").over(window_range), 2)
)

print("\n" + "=" * 60)
print("METHOD 2: 7-Day Rolling Average (rangeBetween):")
print("=" * 60)
result_range.select("store_id", "sale_date", "sales", "rolling_7d_avg_range").show()

# ============================================
# ALTERNATIVE: 7-day rolling average with unbounded preceding
# ============================================
# Some systems use unboundedPreceding to current row

window_alt = Window.partitionBy("store_id") \
                   .orderBy("date_long") \
                   .rangeBetween(-6 * SECONDS_PER_DAY, Window.currentRow)

result_alt = df.withColumn("rolling_avg",
    F.round(F.avg("sales").over(window_alt), 2)
)

print("\n" + "=" * 60)
print("ALTERNATIVE: Using unboundedPreceding:")
print("=" * 60)
result_alt.select("store_id", "sale_date", "sales", "rolling_avg").show()

# ============================================
# ADVANCED: Multiple rolling windows
# ============================================

window_3day = Window.partitionBy("store_id") \
                    .orderBy("sale_date") \
                    .rowsBetween(-2, 0)

window_14day = Window.partitionBy("store_id") \
                     .orderBy("sale_date") \
                     .rowsBetween(-13, 0)

result_multi = df \
    .withColumn("rolling_3d_avg", F.round(F.avg("sales").over(window_3day), 2)) \
    .withColumn("rolling_7d_avg", F.round(F.avg("sales").over(window_rows), 2)) \
    .withColumn("rolling_14d_avg", F.round(F.avg("sales").over(window_14day), 2))

print("\n" + "=" * 60)
print("MULTIPLE ROLLING WINDOWS (3, 7, 14 days):")
print("=" * 60)
result_multi.select(
    "store_id", 
    "sale_date", 
    "sales",
    "rolling_3d_avg",
    "rolling_7d_avg",
    "rolling_14d_avg"
).show()

# ============================================
# EXPECTED OUTPUT (7-Day Rolling Average):
# ============================================
# +--------+----------+-----+-------+------+-------+
# |store_id| sale_date|sales|3d_avg |7d_avg|14d_avg|
# +--------+----------+-----+-------+------+-------+
# |  store1|2024-01-01|1000 | 1000.0| 1000 | 1000  |
# |  store1|2024-01-02|1200 | 1100.0| 1100 | 1100  |
# |  store1|2024-01-03| 900 | 1033.3|1033.3| 1033.3|
# |  store1|2024-01-04|1100 | 1066.7|1066.7| 1066.7|
# |  store1|2024-01-05| 950 | 983.3 |1010.0| 1010.0|
# |  store1|2024-01-06|1300 |1116.7|1065.0| 1065.0|
# |  store1|2024-01-07|1150 |1146.7|1085.7| 1085.7|
# |  store1|2024-01-08|1250 |1233.3|1107.1| 1107.1|
# |  store1|2024-01-09|1100 |1183.3|1114.3| 1114.3|
# +--------+----------+-----+-------+------+-------+

spark.stop()
