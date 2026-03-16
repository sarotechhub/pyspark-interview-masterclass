"""
Q27: Build a session window — group user events within 30-min gaps.

Scenario: Assign a session ID to each user's events.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("Q27_SessionWindow").master("local[*]").getOrCreate()

# ============================================
# SAMPLE DATA
# ============================================
data = [
    (1, "2024-01-01 10:00:00"),
    (1, "2024-01-01 10:05:00"),
    (1, "2024-01-01 10:45:00"),
    (1, "2024-01-01 10:50:00"),
    (2, "2024-01-01 09:00:00"),
    (2, "2024-01-01 09:40:00"),
]
df = spark.createDataFrame(data, ["user_id", "event_time"])
df = df.withColumn("event_time", F.to_timestamp("event_time"))

print("=" * 60)
print("INPUT DATA:")
print("=" * 60)
df.show(truncate=False)

# ============================================
# SOLUTION
# ============================================

window = Window.partitionBy("user_id").orderBy("event_time")

result = df \
    .withColumn("prev_time", F.lag("event_time", 1).over(window)) \
    .withColumn("gap_seconds",
        F.unix_timestamp("event_time") - F.unix_timestamp("prev_time")
    ) \
    .withColumn("new_session_flag",
        F.when(
            F.col("gap_seconds").isNull() |
            (F.col("gap_seconds") > 1800),
            F.lit(1)
        ).otherwise(F.lit(0))
    ) \
    .withColumn("session_id",
        F.concat(
            F.col("user_id"),
            F.lit("_"),
            F.sum("new_session_flag").over(window)
        )
    ) \
    .drop("prev_time", "gap_seconds", "new_session_flag")

print("\n" + "=" * 60)
print("OUTPUT (With Session IDs):")
print("=" * 60)
result.show(truncate=False)

spark.stop()
