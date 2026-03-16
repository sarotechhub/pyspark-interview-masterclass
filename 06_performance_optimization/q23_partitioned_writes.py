"""
Q23: Partitioned Writes — Write output efficiently by date/region

Scenario: Write large output tables partitioned by date and region,
avoiding too many small files and enabling partition pruning.

Key Concepts:
- Repartition before writing
- Write partitionBy for structure
- Control files per partition
- Optimal partition count strategy
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, LongType, IntegerType, StringType, DoubleType
import tempfile
import os

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Q23_Partitioned_Writes") \
    .master("local[*]") \
    .getOrCreate()

print("=" * 80)
print("Q23: Partitioned Writes — Efficient output structure")
print("=" * 80)

# ============================================================================
# Sample Data
# ============================================================================
print("\n--- Creating sample sales data ---")

data = []
for year in [2023, 2024]:
    for month in range(1, 13):
        for region in ["US", "EU", "APAC"]:
            for i in range(10):
                data.append({
                    "order_id": f"{year}{month:02d}{region[0]}{i}",
                    "customer_id": i % 5,
                    "amount": 100.0 + (i * 50),
                    "order_date": f"{year}-{month:02d}-{(i % 28) + 1:02d}",
                    "country_code": region,
                })

df = spark.createDataFrame(data)

print(f"✓ Created {len(data)} sales records")
print("\nOriginal DataFrame:")
df.show(5)

# ============================================================================
# METHOD 1: WRONG WAY — Write without partitioning
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 1: WRONG WAY (no partitioning) ---")
print("=" * 80)

print("\n❌ INEFFICIENT: Write single large partition")
print("""
df.write.mode("overwrite").parquet("/output/sales/")
  Problem:
  - All files in single folder
  - No partition pruning client-side
  - Slow queries that filter by date
  - Impossible to incrementally add data
""")

# ============================================================================
# METHOD 2: RIGHT WAY — Partitioned write
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 2: RIGHT WAY (partitioned write) ---")
print("=" * 80)

output_path = os.path.join(tempfile.gettempdir(), "sales_partitioned")

print("\n✓ Add partition columns if not present:")
df_with_parts = df \
    .withColumn("year", F.year(F.col("order_date"))) \
    .withColumn("month", F.month(F.col("order_date"))) \
    .withColumn("region", F.col("country_code"))

print("Added columns: year, month, region")

print("\n✓ Repartition by partition keys BEFORE writing:")
print("  (This ensures 1 file per partition folder, not thousands of small files)")

df_repartitioned = df_with_parts.repartition("year", "month", "region")
print(f"  Partitions: {df_repartitioned.rdd.getNumPartitions()}")

print("\n✓ Write with partitionBy():")
df_repartitioned.write \
    .mode("overwrite") \
    .partitionBy("year", "month", "region") \
    .parquet(output_path)

print(f"  ✓ Written to: {output_path}")

# ============================================================================
# METHOD 3: Verify partition structure
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 3: Verify partition structure ---")
print("=" * 80)

print("\n✓ Partition folder structure created:")
import os
for root, dirs, files in os.walk(output_path):
    level = root.replace(output_path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    if level < 3:  # Limit depth for readability
        for file in files[:2]:
            print(f'{indent}  {file}')

# ============================================================================
# METHOD 4: Query with partition pruning
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 4: Query with partition pruning ---")
print("=" * 80)

print("\n✓ Read partitioned data:")
df_read = spark.read.parquet(output_path)

print("\n✓ Query specific partition (fast — reads only that folder):")
result = df_read.filter(
    (F.col("year") == 2024) &
    (F.col("month") == 1) &
    (F.col("region") == "US")
)

print(f"  2024-01 US region: {result.count()} rows")
result.show()

# ============================================================================
# METHOD 5: Incremental appends
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 5: Incremental writes (append new data) ---")
print("=" * 80)

print("\n✓ With partitioned structure, appending new month is easy:")

new_data = []
for region in ["US", "EU", "APAC"]:
    for i in range(5):
        new_data.append({
            "order_id": f"202502{region[0]}{i}",
            "customer_id": i,
            "amount": 150.0 + i * 100,
            "order_date": "2025-02-15",
            "country_code": region,
        })

df_new = spark.createDataFrame(new_data)
df_new = df_new \
    .withColumn("year", F.year(F.col("order_date"))) \
    .withColumn("month", F.month(F.col("order_date"))) \
    .withColumn("region", F.col("country_code"))

print(f"  Appending {len(new_data)} new rows for 2025-02")

df_new.repartition("year", "month", "region").write \
    .mode("append") \
    .partitionBy("year", "month", "region") \
    .parquet(output_path)

print("  ✓ New data appended")

# Verify
df_verify = spark.read.parquet(output_path)
print(f"\n  Total rows now: {df_verify.count()}")

# ============================================================================
# METHOD 6: Optimal partition strategy
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 6: Partition strategy guide ---")
print("=" * 80)

strategy = """
✓ CHOOSE PARTITION KEYS:
  1. Frequently filtered columns:
     - Date (year, month, day)
     - Geographic (region, country, city)
     - Category/type
  
  2. High cardinality but manageable:
     - Think: 10-1000 partitions, not millions
     - Each partition = folder = ~100MB-1GB
  
  3. Avoid:
     - User ID (too many partitions)
     - Millisecond timestamps (too many)
     - Continuous float values

✓ EXAMPLE STRUCTURES:

Small dataset (10GB):
  .partitionBy("year", "month")
  → 24-36 partitions

Medium dataset (100GB):
  .partitionBy("year", "month", "category")
  → 1000+ partitions

Large dataset (10TB):
  .partitionBy("year", "month", "region", "store")
  → 10,000+ partitions

✓ REPARTITION BEFORE WRITE:
  df.repartition("year", "month") \\
    .write.partitionBy("year", "month")
  → Ensures 1 file per partition, not 1000s
  → You choose: 1 big file or many small files
"""
print(strategy)

# ============================================================================
# METHOD 7: Multiple partition scenarios
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 7: Multiple partition scenarios ---")
print("=" * 80)

print("\nScenario A: By date only (simple)")
output_a = os.path.join(tempfile.gettempdir(), "sales_by_date")
df_with_parts.repartition("year", "month") \
    .write.mode("overwrite").partitionBy("year", "month") \
    .parquet(output_a)
print("  ✓ Partitioned by year/month")

print("\nScenario B: By date and region (more granular)")
output_b = os.path.join(tempfile.gettempdir(), "sales_by_date_region")
df_with_parts.repartition("year", "month", "region") \
    .write.mode("overwrite").partitionBy("year", "month", "region") \
    .parquet(output_b)
print("  ✓ Partitioned by year/month/region")

print("\nScenario C: By region only (if filtering mostly by region)")
output_c = os.path.join(tempfile.gettempdir(), "sales_by_region")
df_with_parts.repartition("region") \
    .write.mode("overwrite").partitionBy("region") \
    .parquet(output_c)
print("  ✓ Partitioned by region only")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY — Partitioned Writes")
print("=" * 80)
print("""
✓ PARTITION WRITE SYNTAX:

df.write \\
  .mode("overwrite") \\
  .partitionBy("year", "month", "region") \\
  .parquet(path)

✓ BEFORE WRITING, REPARTITION:

df.repartition("year", "month", "region") \\
  .write.partitionBy("year", "month", "region") \\
  .parquet(path)

✓ BENEFITS:
  1. Partition pruning: Read only needed folders
  2. Incremental writes: Append new partitions easily
  3. Query planning: Spark skips irrelevant data
  4. Cleanup: Delete old partitions simply
  5. Scalability: Handle TB+ of data

✓ FILE COUNT CONTROL:
  - 1 repartition per partition key → 1 file
  - No repartition → many small files (bad)
  - Example: 12 months × 3 regions = 36 files

✓ QUERY EXAMPLES:

Read all data:
  spark.read.parquet(path)

Read 2024 only:
  spark.read.parquet(f"{path}/year=2024")

Read 2024-01 US:
  spark.read.parquet(f"{path}/year=2024/month=01/region=US")

Read with filter (partition pruning):
  spark.read.parquet(path) \\
    .filter((F.col("year") == 2024) & (F.col("region") == "US"))

✓ MAINTENANCE:

Delete old partition:
  shutil.rmtree(f"{path}/year=2023")

Add new data daily:
  df_daily.write.mode("append").partitionBy(...).parquet(path)

Check partitions:
  spark.read.parquet(path).select("year", "month", "region").distinct()
""")

print("=" * 80)

# Cleanup
import shutil
for p in [output_path, output_a, output_b, output_c]:
    try:
        shutil.rmtree(p, ignore_errors=True)
    except Exception as e:
        print(f"Error occurred while cleaning up {p}: {e}")
