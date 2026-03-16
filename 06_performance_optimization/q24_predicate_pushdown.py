"""
Q24: Predicate Pushdown — Optimize queries by filtering at source

Scenario: Avoid reading entire table by pushing filters to the data source
(Parquet, JDBC, Hive partitions, etc.)

Key Concepts:
- Predicate pushdown: Filters pushed to data layer
- Partition pruning: Skip reading entire partitions
- Column pruning: Read only needed columns
- Verify with explain()
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, LongType, IntegerType, StringType, DoubleType
import tempfile
import os

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Q24_Predicate_Pushdown") \
    .master("local[*]") \
    .getOrCreate()

print("=" * 80)
print("Q24: Predicate Pushdown — Efficient filtering at source")
print("=" * 80)

# ============================================================================
# Create Sample Partitioned Dataset
# ============================================================================
print("\n--- Creating partitioned sample data ---")

data = []
for year in [2023, 2024, 2025]:
    for month in range(1, 4):  # Just 3 months for demo
        for customer_id in range(1, 6):
            data.append({
                "order_id": f"{year}{month:02d}{customer_id:04d}",
                "customer_id": customer_id,
                "amount": 100.0 + (customer_id * 50),
                "order_date": f"{year}-{month:02d}-15",
                "status": "completed" if customer_id % 2 == 0 else "pending",
            })

df = spark.createDataFrame(data)

# Add year and month columns and write partitioned
df_part = df \
    .withColumn("year", F.year(F.col("order_date"))) \
    .withColumn("month", F.month(F.col("order_date")))

output_path = os.path.join(tempfile.gettempdir(), "orders_pushdown_demo")
df_part.repartition("year", "month") \
    .write.mode("overwrite").partitionBy("year", "month") \
    .parquet(output_path)

print(f"✓ Created partitioned dataset: {output_path}")
print("  Structure: year/2023/month=1, year=2023/month=2, etc.")

# ============================================================================
# METHOD 1: WRONG WAY — Read all, filter in memory
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 1: WRONG WAY (no predicate pushdown) ---")
print("=" * 80)

print("\n❌ INEFFICIENT: Read all data, then filter in memory")

df_all = spark.read.parquet(output_path)

# Without setting up filter predicates properly
result_bad = df_all \
    .filter(F.col("year") == 2024) \
    .filter(F.col("month") == 1)

print(f"  Rows: {result_bad.count()}")

print("\nExplain plan (without pushdown optimization):")
result_bad.explain("formatted")

# ============================================================================
# METHOD 2: RIGHT WAY — Let Spark push filters to Parquet reader
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 2: RIGHT WAY (predicate pushdown) ---")
print("=" * 80)

print("\n✓ EFFICIENT: Place filter WHERE clause before read")

df_opt = spark.read.parquet(output_path) \
    .filter(F.col("year") == 2024) \
    .filter(F.col("month") == 1)

print(f"  Rows: {df_opt.count()}")

print("\nExplain plan (WITH pushdown optimization):")
df_opt.explain("formatted")

# ============================================================================
# METHOD 3: Verify predicate pushdown
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 3: Verify pushdown in explain() output ---")
print("=" * 80)

print("\n✓ Look for 'PushedFilters' in explain output:")
print("""
Good sign: PushedFilters: [IsNotNull(year), EqualTo(year,2024), IsNotNull(month), EqualTo(month,1)]
  → Filters pushed to Parquet reader
  → Reads only month=1 of 2024 folders

Bad sign: No PushedFilters or PartitionFilters
  → Reads all data into memory, then filters
  → Much slower and memory-intensive
""")

# ============================================================================
# METHOD 4: Column pruning (select before filter)
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 4: Column pruning + predicate pushdown ---")
print("=" * 80)

print("\n✓ Combine column selection with filtering:")

df_final = spark.read.parquet(output_path) \
    .filter(F.col("year") == 2024) \
    .select("order_id", "customer_id", "amount")

print(f"  Rows: {df_final.count()}")
print(f"  Columns: {df_final.columns}")

print("\nExplain plan with both optimizations:")
df_final.explain("formatted")

# ============================================================================
# METHOD 5: Direct partition path access
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 5: Direct partition path (fastest) ---")
print("=" * 80)

print("\n✓ FASTEST: Read only the partition folder you need")

specific_path = os.path.join(output_path, "year=2024", "month=1")

if os.path.exists(specific_path):
    df_direct = spark.read.parquet(specific_path)
    print(f"  Rows: {df_direct.count()}")
    print("  ✓ Reads only 1 partition folder (not entire table)")
else:
    print("  (Partition not found in this demo)")

# ============================================================================
# METHOD 6: Avoid anti-patterns
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 6: Anti-patterns that prevent pushdown ---")
print("=" * 80)

print("\n❌ ANTI-PATTERN 1: Filter after transformation")
print("""
df.read.parquet(path) \\
  .withColumn("year_str", F.col("year").cast("string")) \\
  .filter(F.col("year_str") == "2024")   # Uses derived column!
  → Spark can't push this filter to source
  → Reads all data
""")

print("\n❌ ANTI-PATTERN 2: Complex filter conditions")
print("""
df.filter((F.col("amount") > 100) & (F.col("amount") < 200))
  → Can push these filters
  
But:
df.filter(F.col("amount") * 1.1 > 500)
  → Spark may not push (involves transformation)
""")

print("\n✓ GOOD: Simple, direct filters on source columns")
print("""
df.filter(F.col("year") == 2024)        # ✓ Pushable
df.filter(F.col("amount") > 100)        # ✓ Pushable
df.filter(F.col("status") == "completed")  # ✓ Pushable
""")

# ============================================================================
# METHOD 7: Performance comparison
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 7: Performance comparison ---")
print("=" * 80)

import time

print("\n✓ Query 1: Read entire table, filter in memory (SLOW)")
start = time.time()
df1 = spark.read.parquet(output_path)
result1 = df1.filter((F.col("year") == 2024) & (F.col("month") == 1))
result1.count()
time1 = time.time() - start
print(f"  Time: {time1:.4f}s")

print("\n✓ Query 2: Filter before select, let pushdown work (FAST)")
start = time.time()
df2 = spark.read.parquet(output_path)
result2 = df2.filter((F.col("year") == 2024) & (F.col("month") == 1)) \
             .select("order_id", "customer_id")
result2.count()
time2 = time.time() - start
print(f"  Time: {time2:.4f}s")

speedup = time1 / time2 if time2 > 0 else 1
print(f"\n  Speedup: {speedup:.1f}x")

# ============================================================================
# METHOD 8: JDBC predicate pushdown
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 8: JDBC database predicate pushdown ---")
print("=" * 80)

print("""
✓ With JDBC connections, push filters to database:

df = spark.read.jdbc(
    url="jdbc:postgresql://host/db",
    table="orders",
    predicates=[
        "order_date >= '2024-01-01'",
        "order_date <= '2024-12-31'",
        "status = 'completed'",
        "amount > 100"
    ],
    properties={"user": "user", "password": "pass"}
)

Benefits:
  - Database returns only filtered rows
  - Network bandwidth reduced
  - Faster than reading entire table
  - Especially good for large tables

Supported predicates:
  - =, !=, <, >, <=, >=
  - AND, OR
  - LIKE patterns
  - IN lists
""")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY — Predicate Pushdown")
print("=" * 80)
print("""
✓ PREDICATE PUSHDOWN STRATEGIES:

1. PARQUET FILES:
   df.read.parquet(path) \\
     .filter(F.col("partition_col") == value) \\
     .filter(F.col("other_col") > threshold)
   → Spark reads only matching partitions

2. PARTITIONED PATHS:
   spark.read.parquet("/path/year=2024/month=1/")
   → Direct path access (fastest)

3. JDBC DATABASES:
   spark.read.jdbc(..., predicates=[...])
   → Push filters to database

✓ VERIFY PUSHDOWN:
   df.explain("formatted")
   Look for: PushedFilters: [...]

✓ HOW PUSHDOWN WORKS:
   
   WITHOUT: Read(all) → Filter → Transform
   WITH:    Read(filtered) → Transform
   
   Without pushdown: 100GB read → 1GB matches filter
   With pushdown: 1GB read directly (100x faster!)

✓ BEST PRACTICES:
   1. Filter immediately after read
   2. Use simple, direct filters
   3. Filter by partition key when possible
   4. Verify with explain()
   5. Avoid complex expressions in filters
   6. Check that pushdown is working before prod

✓ PERFORMANCE IMPACT:
   Pushdown:     1 GB read × 50 MB/s = 20 sec
   No pushdown:  100 GB read × 50 MB/s = 2000 sec
   Speedup:      100x faster!
""")

print("=" * 80)

# Cleanup
import shutil
try:
    shutil.rmtree(output_path, ignore_errors=True)
except Exception as e:
    print(f"Error occurred while cleaning up {output_path}: {e}")
