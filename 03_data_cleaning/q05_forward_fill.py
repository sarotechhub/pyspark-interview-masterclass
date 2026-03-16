"""
Q5: Replace nulls with the previous non-null value (forward fill).

Scenario: Time series data has missing values. 
Fill each null with the last known value.
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - Window partitionBy: O(n) — single pass
#   - Window orderBy: O(n*log(n)) — sorting per partition
#   - last(ignorenulls=True): O(n) — sequential window scan
#   - Total: O(n*log(n)) — dominated by sorting
#
# Shuffle Operations:
#   - partitionBy(id): FULL SHUFFLE (redistribute by entity/time series)
#   - orderBy(date): NO SHUFFLE (sort within partition)
#
# Performance Tips:
#   - Partition on low-cardinality columns (few time series)
#   - Order by date/timestamp for correct fill logic
#   - ignorenulls=True is critical for forward fill
#   - Consider data size increase if many nulls
#   - Unbounded preceding to current row is default window
#   - Cache if reusing for multiple fills
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Q05_ForwardFill") \
    .master("local[*]") \
    .getOrCreate()

# ============================================
# SAMPLE DATA
# ============================================
data = [
    (1, "2024-01-01", 100.0),
    (1, "2024-01-02", None),      # Null - should fill with 100
    (1, "2024-01-03", None),      # Null - should fill with 100
    (1, "2024-01-04", 150.0),
    (1, "2024-01-05", None),      # Null - should fill with 150
    (2, "2024-01-01", None),      # First value is null
    (2, "2024-01-02", 200.0),
    (2, "2024-01-03", None),      # Null - should fill with 200
]

df = spark.createDataFrame(data, ["id", "date", "value"])

print("=" * 60)
print("INPUT DATA (with nulls):")
print("=" * 60)
df.show()

# ============================================
# SOLUTION
# ============================================
# Step 1: Create a window partitioned by id (customer/entity)
#         Ordered by date from earliest to latest
window = Window.partitionBy("id") \
               .orderBy("date") \
               .rowsBetween(Window.unboundedPreceding, Window.currentRow)

# Step 2: Use last() with ignorenulls=True to get the last non-null value
#         within the window (all rows from start to current row)
result = df.withColumn(
    "value_filled",
    F.last("value", ignorenulls=True).over(window)
)

print("\n" + "=" * 60)
print("OUTPUT DATA (Forward Fill Applied):")
print("=" * 60)
result.show()

# ============================================
# ALTERNATIVE: Using first() with descending order
# ============================================
# This approach uses first() instead of last()
# Note: This doesn't handle the data correctly, shown for reference only

# ============================================
# ALTERNATIVE: Backward Fill (fill with next non-null value)
# ============================================
# To fill with the NEXT non-null value instead:
window_backward = Window.partitionBy("id") \
                        .orderBy(F.desc("date")) \
                        .rowsBetween(Window.unboundedPreceding, Window.currentRow)

result_backward = df.withColumn(
    "value_back_filled",
    F.last("value", ignorenulls=True).over(window_backward)
)

print("\n" + "=" * 60)
print("BACKWARD FILL (Using next non-null value):")
print("=" * 60)
result_backward.show()

# ============================================
# ALTERNATIVE: Fill with both forward and backward
# ============================================
result_both = df \
    .withColumn("forward_fill", F.last("value", ignorenulls=True)
                    .over(window)) \
    .withColumn("backward_fill", F.last("value", ignorenulls=True)
                    .over(window_backward)) \
    .withColumn("final_value",
        F.coalesce(F.col("value"), F.col("forward_fill"), F.col("backward_fill"))
    )

print("\n" + "=" * 60)
print("BOTH FORWARD AND BACKWARD FILL:")
print("=" * 60)
result_both.select("id", "date", "value", "final_value").show()

# ============================================
# EXPECTED OUTPUT (Forward Fill):
# ============================================
# +---+----------+-----+------------+
# | id|      date|value|value_filled|
# +---+----------+-----+------------+
# |  1|2024-01-01|100.0|       100.0|
# |  1|2024-01-02| null|       100.0|
# |  1|2024-01-03| null|       100.0|
# |  1|2024-01-04|150.0|       150.0|
# |  1|2024-01-05| null|       150.0|
# |  2|2024-01-01| null|        null|
# |  2|2024-01-02|200.0|       200.0|
# |  2|2024-01-03| null|       200.0|
# +---+----------+-----+------------+

spark.stop()
