"""
Q12: Find customers who placed orders in consecutive months.

Scenario: Identify customers with continuous purchasing activity.
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - date_trunc() + distinct(): O(n) + shuffle
#   - partitionBy(customer_id): O(n) — redistribute
#   - orderBy(order_month): O(n*log(n)) — sort per partition
#   - months_between(): O(n) — date arithmetic
#   - Total: O(n*log(n) + shuffle) — dominated by sorting + shuffle
#
# Shuffle Operations:
#   - distinct() before window: SHUFFLE (dedup by time)
#   - partitionBy() in window: SHUFFLE (redistribute by customer)
#   - Two shuffle operations total
#
# Performance Tips:
#   - Distinct() removes duplicate months early (reduces data)
#   - Two shuffles unavoidable — data size determines cost
#   - Filter consecutive after window for output
#   - months_between() returns decimal (handle rounding)
#   - months_between(col1, col2) returns col1-col2
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Q12_ConsecutiveMonths") \
    .master("local[*]") \
    .getOrCreate()

# ============================================
# SAMPLE DATA
# ============================================
data = [
    (1, "2024-01-01"),
    (1, "2024-02-15"),  # Consecutive with Jan (same month)
    (1, "2024-04-10"),  # Gap - not consecutive with Feb
    (2, "2024-01-01"),
    (2, "2024-02-01"),
    (2, "2024-03-01"),  # All 3 months consecutive
    (3, "2024-01-15"),
    (3, "2024-01-20"),  # Same month as previous
    (3, "2024-03-10"),  # Gap from January
]

df = spark.createDataFrame(data, ["customer_id", "order_date"])

# Convert to date and extract year-month
df = df.withColumn("order_date", F.to_date("order_date")) \
       .withColumn("order_month", F.date_trunc("month", "order_date")) \
       .distinct()  # Get unique months per customer

print("=" * 60)
print("INPUT DATA (Unique months per customer):")
print("=" * 60)
df.show()

# ============================================
# SOLUTION
# ============================================

# Step 1: Create a window ordered by month, partitioned by customer
window = Window.partitionBy("customer_id").orderBy("order_month")

# Step 2: Use lag() to get the previous month for each customer
result = df \
    .withColumn("prev_month", F.lag("order_month", 1).over(window)) \
    .withColumn("months_gap",
        # months_between calculates the number of months between two dates
        F.months_between(F.col("order_month"), F.col("prev_month"))
    ) \
    .withColumn("is_consecutive",
        # A gap of exactly 1 month means consecutive
        F.when(F.col("months_gap") == 1, True).otherwise(False)
    )

print("\n" + "=" * 60)
print("INTERMEDIATE: Gap Analysis:")
print("=" * 60)
result.show()

# Step 3: Filter to show only consecutive months
result_consecutive = result.filter(F.col("is_consecutive") == True)

print("\n" + "=" * 60)
print("OUTPUT: Customers with consecutive months:")
print("=" * 60)
result_consecutive.select("customer_id", "order_month", "prev_month").show()

# ============================================
# ADVANCED: Find consecutive streaks
# ============================================
# Group consecutive orders into streaks

window_adv = Window.partitionBy("customer_id").orderBy("order_month")

result_streaks = df \
    .withColumn("prev_month", F.lag("order_month", 1).over(window_adv)) \
    .withColumn("months_gap",
        F.months_between(F.col("order_month"), F.col("prev_month"))
    ) \
    .withColumn("gap_flag",
        F.when(F.col("months_gap") == 1 | F.col("months_gap").isNull(), 0)
         .otherwise(1)
    ) \
    .withColumn("streak_id",
        F.sum("gap_flag").over(
            Window.partitionBy("customer_id")
                  .orderBy("order_month")
                  .rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )
    )

# Calculate streak length
streak_length = result_streaks.groupBy("customer_id", "streak_id").agg(
    F.count("*").alias("streak_length"),
    F.min("order_month").alias("streak_start"),
    F.max("order_month").alias("streak_end")
)

print("\n" + "=" * 60)
print("CONSECUTIVE STREAK ANALYSIS:")
print("=" * 60)
streak_length.filter(F.col("streak_length") > 1).show()

# ============================================
# ALTERNATIVE: Find longest consecutive streak per customer
# ============================================
longest_streak = streak_length.groupBy("customer_id").agg(
    F.max("streak_length").alias("longest_streak_months"),
    F.first("streak_start").alias("latest_streak_start")
)

print("\n" + "=" * 60)
print("LONGEST STREAK PER CUSTOMER:")
print("=" * 60)
longest_streak.show()

# ============================================
# EXPECTED OUTPUT (Consecutive months):
# ============================================
# +-----------+----------+----------+
# |customer_id|order_month|prev_month|
# +-----------+----------+----------+
# |          1|2024-02-01|2024-01-01|
# |          2|2024-02-01|2024-01-01|
# |          2|2024-03-01|2024-02-01|
# +-----------+----------+----------+

# ============================================
# EXPECTED OUTPUT (Streaks):
# ============================================
# +-----------+----------+----------+----------+
# |customer_id|streak_len|streak_st |streak_end|
# +-----------+----------+----------+----------+
# |          2|         3|2024-01-01|2024-03-01|
# |          1|         2|2024-01-01|2024-02-01|
# +-----------+----------+----------+----------+

spark.stop()
