"""
Q21: Your Spark job is very slow — how do you diagnose and fix it?

Scenario: Identify performance bottlenecks and apply fixes.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("Q21_DiagnoseSlow") \
    .master("local[*]") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# ============================================
# SAMPLE DATA (Simulated slow operation)
# ============================================
data = [(i, i % 100, i * 10) for i in range(100000)]
df = spark.createDataFrame(data, ["id", "category", "value"])

print("=" * 60)
print("DIAGNOSIS STEPS:")
print("=" * 60)

# ============================================
# STEP 1: Check query plan
# ============================================
print("\n1. QUERY PLAN (basic):")
df.filter(F.col("category") > 50).groupBy("category").agg(F.sum("value")).explain()

# ============================================
# STEP 2: Check partition count
# ============================================
print("\n2. PARTITION COUNT:")
print(f"Total partitions: {df.rdd.getNumPartitions()}")

# Check partition sizes
partition_sizes = df.groupBy(F.spark_partition_id().alias("partition")).count()
print("Partition size distribution:")
partition_sizes.orderBy(F.desc("count")).show()

# ============================================
# STEP 3: Example fixes
# ============================================

# FIX 1: Increase partitions if too few
df_repartitioned = df.repartition(200)

# FIX 2: Decrease partitions (use coalesce - no shuffle)
df_coalesced = df.coalesce(10)

# FIX 3: Cache intermediate results
df_cached = df.filter(F.col("category") > 50).cache()

# FIX 4: Enable adaptive execution (already set above)
result_adaptive = df.filter(F.col("category") > 50) \
    .groupBy("category") \
    .agg(F.sum("value"))

print("\n3. APPLY FIXES AND SHOW RESULTS:")
result_adaptive.show()

spark.stop()
