"""
Q26: Identify churned customers — no purchase in last 90 days.

Scenario: Flag customers with no recent activity for re-engagement campaigns.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("Q26_ChurnDetection").master("local[*]").getOrCreate()

# ============================================
# SAMPLE DATA
# ============================================
data = [
    (1, "2024-01-15"),
    (1, "2024-02-20"),
    (2, "2023-10-01"),
    (2, "2023-11-01"),
    (3, "2024-03-10"),
    (4, "2024-01-05"),
    (5, "2023-08-15"),
]
df = spark.createDataFrame(data, ["customer_id", "purchase_date"])
df = df.withColumn("purchase_date", F.to_date("purchase_date"))

print("=" * 60)
print("INPUT DATA (Customer purchases):")
print("=" * 60)
df.show()

# ============================================
# SOLUTION
# ============================================

# Step 1: Get the most recent purchase date for each customer
last_purchase = df.groupBy("customer_id") \
    .agg(F.max("purchase_date").alias("last_purchase_date"))

# Step 2: Calculate days since last purchase
result = last_purchase \
    .withColumn("days_inactive",
        F.datediff(F.current_date(), F.col("last_purchase_date"))
    ) \
    .withColumn("status",
        F.when(F.col("days_inactive") > 90, "Churned")
         .when(F.col("days_inactive") > 30, "At Risk")
         .otherwise("Active")
    ) \
    .orderBy(F.desc("days_inactive"))

print("\n" + "=" * 60)
print("OUTPUT (Churn Detection):")
print("=" * 60)
result.show()

# ============================================
# SUMMARY REPORT
# ============================================
summary = result.groupBy("status").count()
print("\n" + "=" * 60)
print("CUSTOMER STATUS SUMMARY:")
print("=" * 60)
summary.show()

spark.stop()
