"""
Q14: Assign percentile buckets (quartiles) to customers by spend.

Scenario: Segment customers into quartiles based on their total spending.
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - orderBy(total_spend): O(n*log(n)) — full sort
#   - ntile(k): O(n) — assign buckets after sort
#   - percent_rank(): O(n) — sequential rank calculation
#   - cume_dist(): O(n) — cumulative distribution
#   - Total: O(n*log(n)) — dominated by sorting
#
# Shuffle Operations:
#   - Global orderBy(): SHUFFLE (single partition sort)
#   - Note: Global sort is expensive for large data
#   - Consider partitionBy before orderBy to distribute
#
# Performance Tips:
#   - ntile(4) creates 4 buckets (quartiles) efficiently
#   - percent_rank() returns [0.0, 1.0] range (proportion)
#   - cume_dist() also returns [0.0, 1.0] (cumulative)
#   - Global orderBy expensive — use only if necessary
#   - approx_percentile() for approximate percentiles (faster)
#   - Consider: histogram_approx() for distributions
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Q14_PercentileBuckets") \
    .master("local[*]") \
    .getOrCreate()

# ============================================
# SAMPLE DATA
# ============================================
data = [(i, i * 100) for i in range(1, 21)]
df = spark.createDataFrame(data, ["customer_id", "total_spend"])

print("=" * 60)
print("INPUT DATA (20 customers with varying spend):")
print("=" * 60)
df.show(20)

# ============================================
# SOLUTION
# ============================================

# Create a window ordered by total_spend
window = Window.orderBy("total_spend")

# Step 1: ntile(n) - divides data into n buckets
#         ntile(4) = quartiles (Q1, Q2, Q3, Q4)
#         ntile(10) = deciles
result = df \
    .withColumn("quartile",  F.ntile(4).over(window)) \
    .withColumn("decile",    F.ntile(10).over(window)) \
    .withColumn("percentile", F.percent_rank().over(window)) \
    .withColumn("cum_dist",  F.cume_dist().over(window))

print("\n" + "=" * 60)
print("OUTPUT: Percentile Buckets:")
print("=" * 60)
result.show(20)

# ============================================
# ALTERNATIVE: Using approx_percentile for static thresholds
# ============================================

# Calculate the actual spending threshold for each quartile
quartile_thresholds = df.select([
    F.percentile_approx("total_spend", 0.25).alias("Q1_75th"),
    F.percentile_approx("total_spend", 0.50).alias("Q2_50th"),
    F.percentile_approx("total_spend", 0.75).alias("Q3_25th"),
])

print("\n" + "=" * 60)
print("QUARTILE THRESHOLDS:")
print("=" * 60)
quartile_thresholds.show()

# Get the actual threshold values
q1_val = df.select(F.percentile_approx("total_spend", 0.25)).collect()[0][0]
q2_val = df.select(F.percentile_approx("total_spend", 0.50)).collect()[0][0]
q3_val = df.select(F.percentile_approx("total_spend", 0.75)).collect()[0][0]

# Assign quartiles based on thresholds
result_threshold = df.withColumn("quartile_name",
    F.case_when(
        F.col("total_spend") <= q1_val, "Q1 (Low)"
    ).when(F.col("total_spend") <= q2_val, "Q2 (Medium)"
    ).when(F.col("total_spend") <= q3_val, "Q3 (High)"
    ).otherwise("Q4 (Premium)")
)

print("\n" + "=" * 60)
print("QUARTILE ASSIGNMENT (Using thresholds):")
print("=" * 60)
result_threshold.show(20)

# ============================================
# ADVANCED: Bucket with custom names
# ============================================

# Add descriptive names for each quartile
result_named = df \
    .withColumn("quartile", F.ntile(4).over(window)) \
    .withColumn("quartile_name",
        F.case_when(
            F.col("quartile") == 1, "Low Spenders (0-25%)"
        ).when(F.col("quartile") == 2, "Medium Spenders (25-50%)"
        ).when(F.col("quartile") == 3, "High Spenders (50-75%)"
        ).otherwise("Premium Spenders (75-100%)")
    ) \
    .withColumn("percentile_group",
        F.case_when(
            F.col("percentile") < 0.25, "Bottom 25%"
        ).when(F.col("percentile") < 0.50, "25-50%"
        ).when(F.col("percentile") < 0.75, "50-75%"
        ).otherwise("Top 25%")
    )

print("\n" + "=" * 60)
print("ADVANCED: Named Segments:")
print("=" * 60)
result_named.select(
    "customer_id", 
    "total_spend", 
    "quartile_name", 
    F.round(F.col("percentile") * 100, 2).alias("percentile_%")
).show(20)

# ============================================
# SUMMARY STATISTICS
# ============================================

summary = df.groupBy().agg(
    F.min("total_spend").alias("min_spend"),
    F.percentile_approx("total_spend", 0.25).alias("q1_25th"),
    F.percentile_approx("total_spend", 0.50).alias("q2_50th"),
    F.percentile_approx("total_spend", 0.75).alias("q3_75th"),
    F.max("total_spend").alias("max_spend"),
    F.stddev("total_spend").alias("stddev")
)

print("\n" + "=" * 60)
print("SUMMARY STATISTICS:")
print("=" * 60)
summary.show()

# ============================================
# EXPECTED OUTPUT:
# ============================================
# +----------+---------+---------+-------+--------+----------+-----+
# |cust_id   |spend    |quartile |decile |percent |cum_dist|
# +----------+---------+---------+-------+--------+----------+-----+
# |1         |100      |1        |1      |0.025  |0.05    |
# |2         |200      |1        |1      |0.075  |0.10    |
# |3         |300      |1        |2      |0.125  |0.15    |
# |4         |400      |1        |2      |0.175  |0.20    |
# |5         |500      |1        |3      |0.225  |0.25    |
# |6         |600      |2        |3      |0.275  |0.30    |
# ...
# |16        |1600     |4        |9      |0.825  |0.85    |
# |17        |1700     |4        |9      |0.875  |0.90    |
# |18        |1800     |4        |10     |0.925  |0.95    |
# |19        |1900     |4        |10     |0.975  |1.00    |
# |20        |2000     |4        |10     |1.0    |1.00    |
# +----------+---------+---------+-------+--------+----------+-----+

spark.stop()
