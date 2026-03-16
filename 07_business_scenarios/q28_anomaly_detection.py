"""
Q28: Detect anomalies — flag transactions 3 standard deviations above mean.

Scenario: Flag suspicious transactions per category using Z-score.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("Q28_AnomalyDetection").master("local[*]").getOrCreate()

# ============================================
# SAMPLE DATA
# ============================================
data = [
    ("Electronics", 500),
    ("Electronics", 520),
    ("Electronics", 480),
    ("Electronics", 9999),  # Anomaly!
    ("Clothing", 100),
    ("Clothing", 110),
    ("Clothing", 95),
    ("Clothing", 950),      # Anomaly!
]
df = spark.createDataFrame(data, ["category", "amount"])

print("=" * 60)
print("INPUT DATA:")
print("=" * 60)
df.show()

# ============================================
# SOLUTION
# ============================================

# Calculate mean and stddev per category
stats = df.groupBy("category").agg(
    F.mean("amount").alias("mean_amount"),
    F.stddev("amount").alias("stddev_amount")
)

# Join stats and compute Z-score
result = df.join(stats, on="category") \
    .withColumn("z_score",
        F.abs((F.col("amount") - F.col("mean_amount")) / F.col("stddev_amount"))
    ) \
    .withColumn("is_anomaly", F.col("z_score") > 3) \
    .orderBy(F.desc("z_score"))

print("\n" + "=" * 60)
print("OUTPUT (With Anomaly Flags):")
print("=" * 60)
result.select(
    "category",
    "amount",
    F.round("z_score", 2).alias("z_score"),
    "is_anomaly"
).show()

spark.stop()
