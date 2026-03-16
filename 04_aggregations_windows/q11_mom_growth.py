"""
Q11: Calculate month-over-month (MoM) revenue growth %.

Scenario: Analyze how revenue changes from one month to the next.
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - partitionBy(region): O(n) — redistribute
#   - orderBy(year, month): O(n*log(n)) — sort per partition
#   - lag() / lead(): O(n) — sequential scans within partition
#   - Arithmetic operations: O(n) — per-row calculations
#   - Total: O(n*log(n)) — dominated by sorting
#
# Shuffle Operations:
#   - partitionBy(): SHUFFLE (redistribute by region)
#   - lag(): NO SHUFFLE (sequential within partition)
#   - One shuffle operation total
#
# Performance Tips:
#   - lag()/lead() offset determines lookback (offset=1 is cheapest)
#   - orderBy must include complete time specification
#   - Null handling in percentages (first row often NULL)
#   - when() conditions prevent divide-by-zero errors
#   - Cache before multiple growth calculations
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Q11_MoMGrowth") \
    .master("local[*]") \
    .getOrCreate()

# ============================================
# SAMPLE DATA
# ============================================
data = [
    ("US", 2024, 1, 10000),
    ("US", 2024, 2, 12000),
    ("US", 2024, 3, 11000),
    ("US", 2024, 4, 15000),
    ("EU", 2024, 1, 8000),
    ("EU", 2024, 2, 8500),
    ("EU", 2024, 3, 9200),
    ("EU", 2024, 4, 9500),
]

df = spark.createDataFrame(data, ["region", "year", "month", "revenue"])

print("=" * 60)
print("INPUT DATA:")
print("=" * 60)
df.show()

# ============================================
# SOLUTION
# ============================================

# Step 1: Create a window ordered by year and month
#         partitioned by region
window = Window.partitionBy("region").orderBy("year", "month")

# Step 2: Use lag() to get the previous month's revenue
#         lag(column, offset) returns the value from N rows before
#         lag(column, 1) returns the previous row's value
result = df \
    .withColumn("prev_revenue", F.lag("revenue", 1).over(window)) \
    .withColumn("mom_growth_amount",
        F.col("revenue") - F.col("prev_revenue")
    ) \
    .withColumn("mom_growth_pct",
        F.when(F.col("prev_revenue").isNotNull(),
            F.round(
                (F.col("revenue") - F.col("prev_revenue")) / F.col("prev_revenue") * 100,
                2
            )
        ).otherwise(F.lit(None))
    )

print("\n" + "=" * 60)
print("OUTPUT DATA (MoM Growth Calculation):")
print("=" * 60)
result.show()

# ============================================
# ADVANCED: Year-over-year (YoY) comparison
# ============================================
# Compare the same month in different years

window_yoy = Window.partitionBy("region", "month").orderBy("year")

result_yoy = df \
    .withColumn("prev_year_revenue", F.lag("revenue", 1).over(window_yoy)) \
    .withColumn("yoy_growth_pct",
        F.when(F.col("prev_year_revenue").isNotNull(),
            F.round(
                (F.col("revenue") - F.col("prev_year_revenue")) / F.col("prev_year_revenue") * 100,
                2
            )
        ).otherwise(F.lit(None))
    )

print("\n" + "=" * 60)
print("YEAR-OVER-YEAR (YoY) GROWTH:")
print("=" * 60)
result_yoy.show()

# ============================================
# ADVANCED: Add multiple growth metrics
# ============================================

window_full = Window.partitionBy("region").orderBy("year", "month")

result_full = df \
    .withColumn("prev_month_rev", F.lag("revenue", 1).over(window_full)) \
    .withColumn("next_month_rev", F.lead("revenue", 1).over(window_full)) \
    .withColumn("mom_growth",
        F.when(F.col("prev_month_rev").isNotNull(),
            F.round((F.col("revenue") - F.col("prev_month_rev")) / F.col("prev_month_rev") * 100, 2)
        )
    ) \
    .withColumn("next_month_growth",
        F.when(F.col("next_month_rev").isNotNull(),
            F.round((F.col("next_month_rev") - F.col("revenue")) / F.col("revenue") * 100, 2)
        )
    ) \
    .withColumn("revenue_trend",
        F.when(F.col("mom_growth") > 5, "Strong Growth")
         .when(F.col("mom_growth") > 0, "Growth")
         .when(F.col("mom_growth") < -5, "Decline")
         .otherwise("Stable")
    )

print("\n" + "=" * 60)
print("FULL METRICS (MoM + Direction + Trend):")
print("=" * 60)
result_full.select(
    "region", 
    "year", 
    "month", 
    "revenue", 
    F.col("mom_growth").alias("Mom%"), 
    "revenue_trend"
).show()

# ============================================
# EXPECTED OUTPUT:
# ============================================
# +------+----+-----+-------+----------+----------+
# |region|year|month|revenue|prev_rev |mom_growth|
# +------+----+-----+-------+----------+----------+
# |    US|2024|    1|  10000|      null|      null|
# |    US|2024|    2|  12000|     10000|      20.0|
# |    US|2024|    3|  11000|     12000|      -8.33|
# |    US|2024|    4|  15000|     11000|      36.36|
# |    EU|2024|    1|   8000|      null|      null|
# |    EU|2024|    2|   8500|      8000|       6.25|
# |    EU|2024|    3|   9200|      8500|       8.24|
# |    EU|2024|    4|   9500|      9200|       3.26|
# +------+----+-----+-------+----------+----------+

spark.stop()
