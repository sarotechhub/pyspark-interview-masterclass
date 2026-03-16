"""
Q6: Window Functions Deep Dive — Partition and order for analytics

Scenario: Learn advanced window functions for ranking, aggregation over groups, and time-series.

Key Concepts:
- Window specification (partitionBy, orderBy, rowsBetween, rangeBetween)
- Ranking functions (row_number, rank, dense_rank)
- Aggregation over windows (sum, avg, min, max, lag, lead)
- Frame specification (rows vs range)
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - Window partitionBy alone: O(n) — single pass
#   - Window with orderBy: O(n*log(n)) — sorting within partitions
#   - Ranking functions: O(n*log(n)) — sorted window operations
#   - Lag/lead: O(n) — sequential lookups within partition
#   - Frame aggregates: O(n*k) — k rows in frame per each row
#
# Shuffle Operations:
#   - partitionBy(): SHUFFLE (redistribute by key)
#   - orderBy within window: NO SHUFFLE (sort in partition only)
#   - lag()/lead(): NO SHUFFLE (sequential within partition)
#
# Performance Tips:
#   - Partition on low-cardinality columns (few values)
#   - Use rangeBetween with dates, rowsBetween for fixed counts
#   - Cache dataframe before multiple window operations
#   - Order data BEFORE window operations if possible
#   - Avoid window with very large partitions
#   - Use bounded frames (ROWS BETWEEN) for large windows
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Q06_Window_Functions") \
    .master("local[*]") \
    .getOrCreate()

print("=" * 80)
print("Q6: Window Functions — Advanced Analytics")
print("=" * 80)

# ============================================================================
# Sample Data
# ============================================================================
data = [
    ("sales", "Alice", 2024, 1, 10000),
    ("sales", "Alice", 2024, 2, 12000),
    ("sales", "Bob", 2024, 1, 15000),
    ("sales", "Bob", 2024, 2, 12000),
    ("engineering", "Carol", 2024, 1, 8000),
    ("engineering", "Carol", 2024, 2, 8500),
    ("engineering", "Dave", 2024, 1, 7500),
    ("engineering", "Dave", 2024, 2, 8000),
]

df = spark.createDataFrame(data, ["dept", "name", "year", "month", "revenue"])

print("\nOriginal DataFrame:")
df.show()

# ============================================================================
# METHOD 1: Basic Window — Row Number and Ranking
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 1: Ranking within groups ---")
print("=" * 80)

# Define window: partition by dept, order by revenue descending
window_rank = Window.partitionBy("dept").orderBy(F.desc("revenue"))

print("\n1a. row_number() — Unique rank (1, 2, 3...):")
result = df.withColumn("row_num", F.row_number().over(window_rank))
result.show()

print("\n1b. rank() — Rank with gaps on ties (1, 1, 3...):")
result = df.withColumn("rank", F.rank().over(window_rank))
result.show()

print("\n1c. dense_rank() — Rank without gaps (1, 1, 2...):")
result = df.withColumn("dense_rank", F.dense_rank().over(window_rank))
result.show()

# ============================================================================
# METHOD 2: Lag and Lead — Compare across rows
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 2: Lag and Lead — Row comparison ---")
print("=" * 80)

window_time = Window.partitionBy("name").orderBy("month")

print("\n2a. lag() — Previous row's value:")
result = df.withColumn(
    "prev_month_revenue", 
    F.lag("revenue").over(window_time)
)
result.select("name", "month", "revenue", "prev_month_revenue").show()

print("\n2b. lead() — Next row's value:")
result = df.withColumn(
    "next_month_revenue",
    F.lead("revenue").over(window_time)
)
result.select("name", "month", "revenue", "next_month_revenue").show()

print("\n2c. Calculate growth (revenue - previous revenue):")
result = df.withColumn(
    "prev_revenue",
    F.lag("revenue").over(window_time)
) \
.withColumn(
    "growth",
    F.col("revenue") - F.col("prev_revenue")
)
result.select("name", "month", "revenue", "prev_revenue", "growth").show()

# ============================================================================
# METHOD 3: Aggregate Functions over Windows
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 3: Window aggregations ---")
print("=" * 80)

window_dept = Window.partitionBy("dept")

print("\n3a. Running total (sum all rows from start to current):")
window_running = Window.partitionBy("name").orderBy("month") \
                       .rowsBetween(Window.unboundedPreceding, Window.currentRow)

result = df.withColumn(
    "running_total",
    F.sum("revenue").over(window_running)
)
result.select("name", "month", "revenue", "running_total").show()

print("\n3b. Department statistics:")
result = df.select(
    "name",
    "revenue",
    F.avg("revenue").over(window_dept).alias("dept_avg"),
    F.min("revenue").over(window_dept).alias("dept_min"),
    F.max("revenue").over(window_dept).alias("dept_max")
)
result.show()

print("\n3c. Percentage of department total:")
result = df.withColumn(
    "pct_of_dept",
    F.round(F.col("revenue") / F.sum("revenue").over(window_dept) * 100, 1)
)
result.select("name", "revenue", "pct_of_dept").show()

# ============================================================================
# METHOD 4: Frame Specifications — Rows vs Range
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 4: Frame specifications (rowsBetween vs rangeBetween) ---")
print("=" * 80)

print("\n4a. rowsBetween(unboundedPreceding, currentRow) — All prior rows:")
window_rows = Window.partitionBy("name").orderBy("month") \
                    .rowsBetween(Window.unboundedPreceding, Window.currentRow)

result = df.withColumn(
    "sum_to_date",
    F.sum("revenue").over(window_rows)
)
result.select("name", "month", "revenue", "sum_to_date").show()

print("\n4b. rowsBetween(-1, 0) — Previous row + current row:")
window_2rows = Window.partitionBy("name").orderBy("month") \
                     .rowsBetween(-1, 0)

result = df.withColumn(
    "rolling_2month_avg",
    F.round(F.avg("revenue").over(window_2rows), 0)
)
result.select("name", "month", "revenue", "rolling_2month_avg").show()

# ============================================================================
# METHOD 5: Percentile Functions
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 5: Percentile and ntile functions ---")
print("=" * 80)

window_global = Window.orderBy("revenue")

print("\n5a. ntile(4) — Quartiles:")
result = df.withColumn(
    "quartile",
    F.ntile(4).over(window_global)
)
result.select("name", "revenue", "quartile").show()

print("\n5b. percent_rank() — Percentile rank (0 to 1):")
result = df.withColumn(
    "percentile",
    F.round(F.percent_rank().over(window_global), 2)
)
result.select("name", "revenue", "percentile").show()

print("\n5c. cume_dist() — Cumulative distribution:")
result = df.withColumn(
    "cume_dist",
    F.round(F.cume_dist().over(window_global), 2)
)
result.select("name", "revenue", "cume_dist").show()

# ============================================================================
# METHOD 6: First and Last in Window
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 6: First and Last values in window ---")
print("=" * 80)

window_dept_rev = Window.partitionBy("dept").orderBy("revenue")

print("\n6a. first() — First value in window:")
result = df.withColumn(
    "min_dept_revenue",
    F.first("revenue").over(window_dept_rev)
)
result.select("dept", "name", "revenue", "min_dept_revenue").show()

print("\n6b. last() — Last value in window:")
result = df.withColumn(
    "max_dept_revenue",
    F.last("revenue").over(window_dept_rev)
)
result.select("dept", "name", "revenue", "max_dept_revenue").show()

# ============================================================================
# METHOD 7: Complex Example — Top performer per department
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 7: Complex Example — Top performer per department ---")
print("=" * 80)

window_dept_rank = Window.partitionBy("dept").orderBy(F.desc("revenue"))

result = df \
    .withColumn("rank", F.dense_rank().over(window_dept_rank)) \
    .filter(F.col("rank") == 1) \
    .select("dept", "name", "revenue")

print("\nTop revenue earner per department:")
result.show()

# ============================================================================
# METHOD 8: Session Window Example (time-based)
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 8: Time-based window operations ---")
print("=" * 80)

# Create sample with timestamps
data_time = [
    (1, "2024-01-01", 100),
    (1, "2024-01-02", 150),
    (1, "2024-01-03", 120),
    (1, "2024-01-10", 200),  # Gap > 5 days
    (1, "2024-01-11", 180),
]
df_time = spark.createDataFrame(data_time, ["id", "date", "amount"])

window_all = Window.partitionBy("id").orderBy("date")

result = df_time \
    .withColumn("date", F.to_date(F.col("date"))) \
    .withColumn("prev_date", F.lag("date").over(window_all)) \
    .withColumn(
        "days_since_prev",
        F.datediff(F.col("date"), F.col("prev_date"))
    ) \
    .select("id", "date", "amount", "days_since_prev")

print("\nTime-based analysis:")
result.show()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY — Window Functions")
print("=" * 80)
print("""
✓ DEFINE WINDOW:
  window = Window.partitionBy("col1").orderBy("col2")
  window = Window.partitionBy("col1", "col2").orderBy(desc("col3"))

✓ RANKING FUNCTIONS:
  F.row_number()   → Unique rank (1, 2, 3...)
  F.rank()         → With gaps on ties (1, 1, 3...)
  F.dense_rank()   → No gaps (1, 1, 2...)

✓ ROW COMPARISON:
  F.lag("col", n)  → Previous n rows
  F.lead("col", n) → Next n rows

✓ AGGREGATE FUNCTIONS (over window):
  F.sum(), F.avg(), F.min(), F.max(), F.count()
  F.first(), F.last()
  F.stddev(), F.variance()

✓ PERCENTILE FUNCTIONS:
  F.ntile(4)        → Quartiles (1-4)
  F.percent_rank()  → Percentile (0-1)
  F.cume_dist()     → Cumulative distribution

✓ FRAME SPECIFICATIONS:
  rowsBetween(Window.unboundedPreceding, Window.currentRow)
  rangeBetween(-2, 0)
  - rowsBetween: Count rows relative to current
  - rangeBetween: Range of values relative to current

✓ COMMON PATTERNS:
  - Ranking: Dense_rank + filter(rank == 1)
  - Running total: rowsBetween(unboundedPreceding, currentRow)
  - Moving average: rowsBetween(-2, 2)
  - YoY comparison: lag("col", 12)
  - Top N per group: dense_rank + filter(rank <= N)

✓ PERFORMANCE TIPS:
  - Partition on high-cardinality column first
  - Order by indexed columns if possible
  - Avoid rowsBetween UNBOUNDED FOLLOWING (shuffles all)
  - Use rangeBetween for date-based operations
""")

print("=" * 80)
