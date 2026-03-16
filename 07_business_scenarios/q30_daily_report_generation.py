"""
Q30: Generate a daily report — active users, revenue, top product per day.

Scenario: Create multi-level aggregation report.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("Q30_DailyReport").master("local[*]").getOrCreate()

# ============================================
# SAMPLE DATA
# ============================================
data = [
    ("2024-01-01", "user1", "TV",     5000),
    ("2024-01-01", "user2", "Phone",  3000),
    ("2024-01-01", "user1", "Laptop", 7000),
    ("2024-01-02", "user3", "TV",     4500),
    ("2024-01-02", "user4", "Phone",  3200),
    ("2024-01-02", "user3", "Laptop", 8000),
    ("2024-01-02", "user5", "TV",     5200),
]
df = spark.createDataFrame(data, ["sale_date", "user_id", "product", "amount"])

print("=" * 60)
print("INPUT DATA (Transaction log):")
print("=" * 60)
df.show()

# ============================================
# SOLUTION
# ============================================

# Daily revenue and active users per product
daily_product = df.groupBy("sale_date", "product").agg(
    F.countDistinct("user_id").alias("active_users"),
    F.sum("amount").alias("daily_revenue")
)

# Rank products per day by revenue
window = Window.partitionBy("sale_date").orderBy(F.desc("daily_revenue"))

daily_report = daily_product \
    .withColumn("rank", F.row_number().over(window)) \
    .withColumn("is_top_product", F.col("rank") == 1) \
    .orderBy("sale_date", "rank")

print("\n" + "=" * 60)
print("DAILY PRODUCT REPORT:")
print("=" * 60)
daily_report.select("sale_date", "rank", "product", "daily_revenue", "active_users", "is_top_product").show()

# Overall daily summary
daily_summary = df.groupBy("sale_date").agg(
    F.countDistinct("user_id").alias("total_active_users"),
    F.sum("amount").alias("total_revenue"),
    F.count("*").alias("total_transactions"),
    F.countDistinct("product").alias("products_sold")
)

print("\n" + "=" * 60)
print("DAILY SUMMARY:")
print("=" * 60)
daily_summary.orderBy("sale_date").show()

spark.stop()
