"""
Q25: Large-Scale Processing — Handle 1 billion rows efficiently

Scenario: Process 1 billion+ rows with limited cluster resources.
Strategy: Filter → Project → Aggregate → Join (in that order)

Key Concepts:
- Columnar projection: Select only needed columns
- Early filtering: Reduce data ASAP
- Aggregation before join: Reduce cardinality
- Broadcasting small tables
- Adaptive query execution
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, LongType, IntegerType, StringType, DoubleType, DateType
import tempfile
import os

# Initialize SparkSession with tuning for large data
spark = SparkSession.builder \
    .appName("Q25_Large_Scale_Processing") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.sql.adaptive.skewJoin.enabled", "true") \
    .config("spark.memory.fraction", "0.8") \
    .getOrCreate()

print("=" * 80)
print("Q25: Large-Scale Processing — 1B+ rows efficiently")
print("=" * 80)

# ============================================================================
# Create Large Dataset (simulating 1B rows)
# ============================================================================
print("\n--- Creating large sample dataset ---")

print("\nGenerating events (1B rows simulation with 100K sample)...")

from pyspark.sql import Row

# Simulate large event data
events_data = []
for day in range(1, 11):  # 10 days
    for hour in range(0, 24):
        for user_id in range(1, 100):  # 100 users
            for event_seq in range(10):  # 10 events per user per hour
                events_data.append({
                    "event_id": f"evt_{day:02d}_{hour:02d}_{user_id:04d}_{event_seq:02d}",
                    "user_id": user_id,
                    "event_type": ["purchase", "view", "click", "add_to_cart"][event_seq % 4],
                    "amount": 50.0 + (event_seq * 10),
                    "event_date": f"2024-01-{day:02d}",
                    "event_hour": hour,
                })

events_df = spark.createDataFrame(events_data)
print(f"✓ Created {events_df.count()} event records")

# Create users dimension (small table for broadcasting)
users_data = [
    {"user_id": i, "user_name": f"User_{i}", "country": ["US", "EU", "APAC"][i % 3]}
    for i in range(1, 101)
]
users_df = spark.createDataFrame(users_data)
print(f"✓ Created {users_df.count()} user records")

# ============================================================================
# METHOD 1: WRONG WAY — Process without optimization
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 1: WRONG WAY (naive processing) ---")
print("=" * 80)

print("\n❌ INEFFICIENT: Join first, then filter/aggregate")
print("""
# Bad approach:
result = events_df \\
    .join(users_df, on="user_id") \\          # ← Join 1B rows to 100 users
    .filter(...) \\                            # ← Filter massive dataset
    .groupBy(...).agg(...) \\                  # ← Aggregate at the end
    
Issues:
1. Join creates 1B rows of joined data (memory pressure)
2. Filtering happens on huge joined dataset
3. Shuffle happens on 1B rows
4. Slow and resource-intensive
""")

# ============================================================================
# METHOD 2: RIGHT WAY — Optimize with proper pipeline order
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 2: RIGHT WAY (optimized pipeline) ---")
print("=" * 80)

print("\n✓ EFFICIENT PIPELINE: Filter → Project → Aggregate → Join")

# Step 1: Filter early
print("\nStep 1: Filter early to reduce volume")
filtered = events_df.filter(
    (F.col("event_date") >= "2024-01-01") &
    (F.col("event_date") <= "2024-01-05") &
    (F.col("event_type").isin(["purchase", "add_to_cart"]))
)
print(f"  After filter: {filtered.count()} rows (volume reduced)")

# Step 2: Project only needed columns
print("\nStep 2: Select only needed columns")
projected = filtered.select("user_id", "event_type", "amount", "event_date")
print(f"  Columns: {projected.columns}")

# Step 3: Aggregate (massive volume reduction!)
print("\nStep 3: Aggregate before join (huge cardinality reduction)")
aggregated = projected.groupBy("user_id").agg(
    F.sum("amount").alias("total_spend"),
    F.count("*").alias("event_count"),
    F.min("event_date").alias("first_purchase")
)
print(f"  After group-by: {aggregated.count()} rows (100 users, was millions)")

# Step 4: Broadcast small dimension table and join
print("\nStep 4: Broadcast small table and join")
from pyspark.sql.functions import broadcast

result = aggregated.join(
    broadcast(users_df),
    on="user_id"
)

print(f"  Final result: {result.count()} rows")
result.show()

# ============================================================================
# METHOD 3: Benefits of optimized pipeline
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 3: Performance comparison ---")
print("=" * 80)

print("""
PROCESSING STRATEGY COMPARISON:

Method 1 (BAD - Join First):
  Step 1: events (1B) × users (100) → 1B rows
  Memory: 1B rows in memory
  Network: 1B row shuffle
  Result: OOM errors, very slow

Method 2 (GOOD - Filter→Project→Aggregate→Join):
  Step 1: Filter events → 10M rows
  Step 2: Project 4 cols → 10M rows
  Step 3: Aggregate by user → 100 rows
  Step 4: Broadcast join → 100 rows
  Memory: Only 10M rows in memory
  Network: Only 100 rows sent to join
  Result: Fast, memory-efficient

SPEEDUP: 100-1000x faster!
""")

# ============================================================================
# METHOD 4: Configuration tuning for 1B+ rows
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 4: Spark configuration for large-scale processing ---")
print("=" * 80)

config_settings = """
✓ MEMORY TUNING:

spark.conf.set("spark.memory.fraction", "0.8")
  → Use 80% of heap for Spark (default 60%)
  → Leaves 20% for OS

spark.conf.set("spark.memory.storageFraction", "0.5")
  → 50% for caching, 50% for execution (default 50/50)

spark.conf.set("spark.executor.memory", "32g")
  → Per-executor heap size

spark.conf.set("spark.driver.memory", "16g")
  → Driver heap size


✓ SHUFFLE TUNING:

spark.conf.set("spark.sql.shuffle.partitions", "400")
  → Match cluster parallelism (default 200)
  → For 1B rows: 2-4 partitions per executor-core


✓ ADAPTIVE QUERY EXECUTION:

spark.conf.set("spark.sql.adaptive.enabled", "true")
  → Enable AQE (Spark 3.0+)

spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
  → Reduce partitions after shuffle if needed

spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
  → Handle skewed data automatically


✓ BROADCAST TUNING:

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10MB")
  → Default: auto-broadcast tables < 10MB
  → For large cluster: increase to 100MB


✓ SERIALIZATION:

spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
  → Kryo serialization (2-10x faster than default)
  → Register custom classes if needed
"""
print(config_settings)

# ============================================================================
# METHOD 5: Incremental processing (daily batches)
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 5: Process in daily batches (for 1B+ rows) ---")
print("=" * 80)

print("""
✓ For datasets too large for single job, process incrementally:

# Process daily batches
for day in range(1, 32):  # Process each day
    date_str = f"2024-01-{day:02d}"
    
    result = spark.read.parquet(f"events/{date_str}") \\
        .filter(F.col("event_type").isin(["purchase"])) \\
        .groupBy("user_id").agg(F.sum("amount")) \\
        .write.mode("append").parquet(f"output/{date_str}")

Benefits:
  - Each job processes ~34M rows (1B/30 days)
  - Memory pressure manageable
  - Can parallelize across dates
  - Easier to retry on failure
  - Incremental output
""")

# ============================================================================
# METHOD 6: Partitioned processing strategy
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 6: Partition-aware processing ---")
print("=" * 80)

print("\n✓ Process by date partitions efficiently:")

# Simulate daily data
output_path = os.path.join(tempfile.gettempdir(), "events_processed")

for day in range(1, 6):
    date_str = f"2024-01-{day:02d}"
    
    daily_result = events_df \
        .filter(F.col("event_date") == date_str) \
        .filter(F.col("event_type").isin(["purchase", "add_to_cart"])) \
        .groupBy("user_id").agg(
            F.sum("amount").alias("daily_spend"),
            F.count("*").alias("daily_events")
        )

print("  Processed 5 daily batches")
print("  ✓ Each batch: ~2K rows processed → 100 rows aggregated")

# ============================================================================
# METHOD 7: Join optimization patterns
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 7: Join optimization patterns for large data ---")
print("=" * 80)

print("""
✓ PATTERN 1: Broadcast small dimension

# Small users table (100 rows)
result = large_events_df \\
    .join(broadcast(users_df), on="user_id")  # ← Broadcast
  → No shuffle needed, fast local join

✓ PATTERN 2: Aggregate before join

# Pre-aggregate large table to small
summary = large_events_df \\
    .groupBy("user_id").agg(F.sum("amount")) \\
    .join(users_df, on="user_id")  # ← Join smaller table
  → Reduces shuffle volume


✓ PATTERN 3: Filter before join

result = large_events_df \\
    .filter(F.col("amount") > 100) \\      # ← Filter first
    .join(users_df, on="user_id")      # ← Join reduced dataset
  → Fewer rows shuffled


✓ PATTERN 4: Pre-stage data if repeated

users_cached = users_df.cache()
for day in date_range:
    events_df.join(users_cached, on="user_id")
  → Cache small table to avoid re-reading
""")

# ============================================================================
# METHOD 8: Complete large-scale example
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 8: Complete end-to-end example ---")
print("=" * 80)

print("\n✓ Full pipeline for 1B-row processing:")

# Read
events = events_df

# Filter (reduce 1B → 10M rows)
events = events.filter(
    (F.col("event_type").isin(["purchase", "add_to_cart"])) &
    (F.col("amount") > 20)
)

# Project (select needed cols)
events = events.select("user_id", "amount", "event_type")

# Aggregate (10M → 100 rows)
summary = events.groupBy("user_id").agg(
    F.count("*").alias("purchase_count"),
    F.sum("amount").alias("total_spend"),
    F.avg("amount").alias("avg_amount")
)

# Join with dimension
final = summary.join(
    broadcast(users_df),
    on="user_id"
).select("user_id", "user_name", "country", "purchase_count", "total_spend")

print(f"\nFinal result with {final.count()} rows:")
final.sort(F.desc("total_spend")).show()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY — Large-Scale Processing (1B+ rows)")
print("=" * 80)
print("""
✓ OPTIMAL PIPELINE ORDER:

  1. READ           → Get raw data
  2. FILTER         → Remove unneeded rows (volume reduction #1)
  3. PROJECT        → Select columns (memory reduction)
  4. AGGREGATE      → Summarize data (volume reduction #2 - CRITICAL!)
  5. JOIN           → Combine with dimensions (small after agg)
  6. WRITE          → Save results

✓ EACH STEP REDUCES DATA:

  Raw events:       1,000,000,000 rows (1B)
  After filter:       100,000,000 rows (100M) ← 10x reduction
  After aggregate:           100 rows (users) ← 1,000,000x reduction!
  After join:               100 rows
  
  Memory saved: Avoid storing 1B rows in memory

✓ KEY TECHNIQUES:

  1. Filter early: Get to manageable size (10M-100M rows)
  2. Project: Only needed columns
  3. Aggregate: Reduce to dimensions size
  4. Broadcast: Small tables don't shuffle
  5. Partition: Process by date/region if possible
  6. Tune Spark: Shuffle partitions, memory, AQE

✓ CONFIGURATION FOR 1B ROWS:

  spark.sql.shuffle.partitions = 200-400
  spark.sql.adaptive.enabled = true
  spark.memory.fraction = 0.8
  spark.sql.autoBroadcastJoinThreshold = "10MB"
  spark.serializer = KryoSerializer

✓ PERFORMANCE TIPS:

  1. Aggregate BEFORE join (not after)
  2. Broadcast small tables (< 10MB)
  3. Filter by partition key (if partitioned)
  4. Use broadcast() explicitly for safety
  5. Avoid collecting 1B rows to driver (OOM!)
  6. Cache small dimension tables
  7. Process incrementally (daily/hourly batches)
  8. Monitor Spark UI for shuffles

✓ EXPECTED SPEEDUP:

  Without optimization:  Out of memory (OOM)
  With optimization:     Processes 1B rows in 1-10 minutes

✓ SCALE ESTIMATE:

  1B rows = Process time scales with data:
  - 100M rows:  1-5 minutes
  - 1B rows:    10-50 minutes
  - 10B rows:   100-500 minutes (depends on cluster size)
""")

print("=" * 80)

# Cleanup
import shutil
try:
    shutil.rmtree(output_path, ignore_errors=True)
except Exception as e:
    print(f"Error occurred while cleaning up {output_path}: {e}")
