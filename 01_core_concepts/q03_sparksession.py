"""
Q3: SparkSession Configuration — Initialize and configure Spark

Scenario: Learn how to create and configure SparkSession with different settings.
SparkSession is the entry point for all Spark functionality.

Key Concepts:
- SparkSession builder pattern
- Configuration settings (memory, partitions, serialization, etc.)
- Working with different storage backends (local, HDFS, S3, etc.)
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - SparkSession creation: O(1) — initialization overhead
#   - createDataFrame(): O(n) — distributes data to partitions
#   - SQL queries: Depends on query complexity
#   - getOrCreate(): O(1) — reuses existing session
#
# Shuffle Operations:
#   - No shuffles in session creation or configuration
#   - Shuffles occur during actual data operations (groupBy, join, etc.)
#
# Performance Tips:
#   - Create session ONCE at application start
#   - Use getOrCreate() in notebooks/interactive environments
#   - Configure driver memory high enough for broadcast joins
#   - Set executor memory based on data size
#   - Tune shufflePartitions based on cluster size
#   - Enable dynamic allocation for variable workloads
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types

# ============================================================================
# METHOD 1: Basic SparkSession
# ============================================================================
print("=" * 80)
print("Q3: SparkSession Configuration")
print("=" * 80)

print("\n--- METHOD 1: Basic SparkSession ---")

# Already created at top of this file
spark = SparkSession.builder \
    .appName("Q03_SparkSession_Config") \
    .master("local[*]") \
    .getOrCreate()

print(f"✓ Spark Version: {spark.version}")
print(f"✓ App Name: {spark.sparkContext.appName}")
print(f"✓ Master: {spark.sparkContext.master()}")

# ============================================================================
# METHOD 2: Advanced Configuration
# ============================================================================
print("\n--- METHOD 2: Advanced Configuration ---")

spark2 = SparkSession.builder \
    .appName("Q03_Advanced_Config") \
    .master("local[4]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "4") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.sql.autoBroadcastJoinThreshold", "10mb") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .getOrCreate()

print("✓ Configuration set for:")
print("  - Driver memory: 4g")
print("  - Executor memory: 4g")
print("  - Executor cores: 4")
print("  - Shuffle partitions: 200")
print("  - Broadcast threshold: 10MB")
print("  - Serializer: Kryo (fast)")

# ============================================================================
# METHOD 3: Configuration Options Reference
# ============================================================================
print("\n--- METHOD 3: Common Configuration Options ---")

config_ref = {
    "Memory": {
        "spark.driver.memory": "Driver process heap size (e.g., '4g', '8g')",
        "spark.executor.memory": "Executor JVM heap size (e.g., '4g')",
        "spark.memory.fraction": "Spark memory as fraction of heap (default 0.6)",
        "spark.memory.storageFraction": "Storage memory fraction (default 0.5)",
    },
    "Cores": {
        "spark.executor.cores": "Cores per executor (e.g., 4)",
        "spark.driver.cores": "Driver cores (for cluster mode)",
    },
    "Shuffle": {
        "spark.sql.shuffle.partitions": "Default shuffle partitions (default 200)",
    },
    "Broadcast": {
        "spark.sql.autoBroadcastJoinThreshold": "Max size for broadcast join (default 10MB)",
    },
    "Serialization": {
        "spark.serializer": "Use KryoSerializer for performance",
    },
    "Adaptive Query Execution": {
        "spark.sql.adaptive.enabled": "Enable adaptive execution (true/false)",
        "spark.sql.adaptive.coalescePartitions.enabled": "Coalesce final partitions",
        "spark.sql.adaptive.skewJoin.enabled": "Handle skewed joins",
    }
}

for category, configs in config_ref.items():
    print(f"\n{category}:")
    for key, desc in configs.items():
        print(f"  {key}")
        print(f"    → {desc}")

# ============================================================================
# METHOD 4: Access Configuration
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 4: Access Current Configuration ---")
print("=" * 80)

# Get config value
shuffle_partitions = spark.conf.get("spark.sql.shuffle.partitions")
print(f"\nCurrent shuffle partitions: {shuffle_partitions}")

# Get all configs (many)
print(f"\nTotal configs available: {len(spark.sparkContext.getConf().getAll())}")

# Print some key configs
print("\nKey Runtime Configs:")
for key, value in spark.sparkContext.getConf().getAll():
    if any(x in key for x in ["shuffle", "broadcast", "memory", "executor"]):
        print(f"  {key}: {value}")

# ============================================================================
# METHOD 5: Set Configuration at Runtime
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 5: Set Configuration at Runtime ---")
print("=" * 80)

# Can modify before first action
print("\nSetting shuffle partitions to 100:")
spark.conf.set("spark.sql.shuffle.partitions", "100")
current = spark.conf.get("spark.sql.shuffle.partitions")
print(f"  → Current value: {current}")

# Reset to default
spark.conf.set("spark.sql.shuffle.partitions", "200")

# ============================================================================
# METHOD 6: SparkContext and SparkSession relationship
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 6: SparkSession vs SparkContext ---")
print("=" * 80)

print("\nSparkSession details:")
print(f"  - Version: {spark.version}")
print(f"  - AppName: {spark.appName}")

print("\nSparkContext (sc) details:")
sc = spark.sparkContext
print(f"  - Master: {sc.master()}")
print(f"  - App Name: {sc.appName}")
print(f"  - Status: {sc.statusTracker().getExecutorInfos()}")

# ============================================================================
# METHOD 7: Create Data with SparkSession
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 7: Create and Work with Data ---")
print("=" * 80)

# From Python list
print("\n1. CreateDataFrame from list:")
df = spark.createDataFrame([
    (1, "Alice", 30),
    (2, "Bob", 25),
    (3, "Carol", 35),
], ["id", "name", "age"])
df.show()

# From Pandas
print("\n2. Create from Pandas DataFrame:")
try:
    import pandas as pd
    pdf = pd.DataFrame({
        "id": [1, 2],
        "value": [100, 200]
    })
    pdf_spark = spark.createDataFrame(pdf)
    pdf_spark.show()
except ImportError:
    print("  Pandas not installed")

# From SQL
print("\n3. Create from SQL:")
spark.sql("SELECT 1 as id, 'Alice' as name, 30 as age").show()

# From file (example structure)
print("\n4. Read from file:")
print("  df = spark.read.csv('path/to/file.csv', header=True)")
print("  df = spark.read.parquet('path/to/file.parquet')")
print("  df = spark.read.json('path/to/file.json')")

# ============================================================================
# METHOD 8: Temp Views for SQL
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 8: SQL Temp Views ---")
print("=" * 80)

# Register temp view
df.createOrReplaceTempView("people")
print("\n✓ Registered temp view 'people'")

# Query using SQL
result = spark.sql("SELECT name, age FROM people WHERE age > 25")
print("\nSQL Query: SELECT name, age FROM people WHERE age > 25")
result.show()

# ============================================================================
# METHOD 9: Read/Write Modes
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 9: Write Modes ---")
print("=" * 80)

write_modes = {
    "overwrite": "Overwrite existing data",
    "append": "Append to existing data",
    "ignore": "Ignore if data already exists",
    "error": "Throw error if data exists (default)",
}

print("\nAvailable write modes:")
for mode, desc in write_modes.items():
    print(f"  {mode}: {desc}")

print("\nExample:")
print("  df.write.mode('overwrite').parquet('/tmp/output')")

# ============================================================================
# METHOD 10: Performance Tuning Settings
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 10: Performance Tuning Configuration ---")
print("=" * 80)

performance_config = """
# For Large Data Processing:
spark.conf.set("spark.sql.shuffle.partitions", "400")
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

# For Memory-constrained clusters:
spark.conf.set("spark.memory.fraction", "0.8")
spark.conf.set("spark.memory.storageFraction", "0.5")

# For Better Serialization:
spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

# For Dynamic Allocation:
spark.conf.set("spark.dynamicAllocation.enabled", "true")
spark.conf.set("spark.dynamicAllocation.minExecutors", "2")
spark.conf.set("spark.dynamicAllocation.maxExecutors", "20")
"""

print(performance_config)

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("SUMMARY — SparkSession Configuration")
print("=" * 80)
print("""
✓ CREATE SPARKSESSION:
  spark = SparkSession.builder \\
      .appName("MyApp") \\
      .master("local[*]") \\
      .config("key", "value") \\
      .getOrCreate()

✓ COMMON CONFIGS:
  - spark.driver.memory: Driver heap size
  - spark.executor.memory: Executor heap size
  - spark.sql.shuffle.partitions: Shuffle partitions (200 default)
  - spark.sql.autoBroadcastJoinThreshold: Broadcast threshold (10MB default)
  - spark.serializer: Serialization (use KryoSerializer)

✓ MASTER OPTIONS:
  - "local[*]": Local mode, use all cores
  - "local[4]": Local mode, use 4 cores
  - "yarn": Run on Hadoop YARN
  - "k8s://kubernetes-cluster": Kubernetes
  - "mesos://mesos-cluster": Mesos cluster

✓ SET CONFIG:
  spark.conf.set("key", "value")

✓ ACCESS CONFIG:
  value = spark.conf.get("key")

✓ BEST PRACTICES:
  1. Use SparkSession (replaces SparkContext + SQLContext)
  2. Configure memory based on cluster/machine
  3. Tune shuffle partitions (default 200)
  4. Use Kryo serialization for performance
  5. Enable adaptive query execution (Spark 3+)
""")

print("=" * 80)

# Cleanup
spark.stop()
print("\n✓ SparkSession stopped")
