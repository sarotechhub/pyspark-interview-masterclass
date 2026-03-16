# 🔥 PySpark Interview Preparation Guide
> **Complete Reference** — Top 30 Theory Questions + Top 30 Scenario-Based Coding Questions  
> Optimized for VS Code + GitHub Copilot

---

## 📁 Recommended Repo Structure

```
pyspark-interview-prep/
├── README.md
├── 01_core_concepts/
│   ├── 01_rdd_basics.py
│   ├── 02_transformations_actions.py
│   └── 03_sparksession.py
├── 02_dataframes_sql/
│   ├── 04_select_withcolumn.py
│   ├── 05_null_handling.py
│   └── 06_window_functions.py
├── 03_data_cleaning/
│   ├── 07_deduplication.py
│   ├── 08_flatten_json.py
│   └── 09_date_parsing.py
├── 04_aggregations_windows/
│   ├── 10_running_total.py
│   ├── 11_top_n_per_group.py
│   └── 12_rolling_average.py
├── 05_joins_set_ops/
│   ├── 13_anti_join.py
│   ├── 14_broadcast_join.py
│   └── 15_skewed_join.py
├── 06_performance_optimization/
│   ├── 16_repartition_coalesce.py
│   ├── 17_cache_persist.py
│   └── 18_predicate_pushdown.py
└── 07_business_scenarios/
    ├── 19_churn_detection.py
    ├── 20_session_window.py
    └── 21_anomaly_detection.py
```

---

## 🧠 PART 1 — Top 30 Theory Interview Questions

---

### 🔷 Section 1: Core Concepts (Q1–Q8)

---

#### Q1. What is Apache Spark and how does PySpark relate to it?

**Answer:**  
Apache Spark is an open-source, distributed computing engine designed for large-scale data processing. It provides in-memory computation, making it significantly faster than Hadoop MapReduce.

PySpark is the **Python API for Apache Spark**. It allows Python developers to write Spark jobs using Python syntax while leveraging Spark's distributed processing power under the hood.

```python
# Basic PySpark setup
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("MyFirstApp") \
    .master("local[*]") \
    .getOrCreate()

sc = spark.sparkContext
print(sc.version)  # Check Spark version
```

---

#### Q2. What is an RDD?

**Answer:**  
RDD (Resilient Distributed Dataset) is the **fundamental data structure** of Apache Spark.

| Property     | Description                                          |
|--------------|------------------------------------------------------|
| Resilient    | Fault-tolerant via lineage graph                     |
| Distributed  | Data split across multiple nodes in the cluster      |
| Dataset      | Collection of partitioned data                       |

**Two types of operations:**
- **Transformations** — Lazy, return a new RDD (`map`, `filter`, `flatMap`)
- **Actions** — Trigger execution, return results (`collect`, `count`, `saveAsTextFile`)

```python
# Create RDD
rdd = sc.parallelize([1, 2, 3, 4, 5])

# Transformation (lazy)
squared = rdd.map(lambda x: x ** 2)

# Action (triggers execution)
print(squared.collect())  # [1, 4, 9, 16, 25]
```

---

#### Q3. What is the difference between RDD, DataFrame, and Dataset?

**Answer:**

| Feature        | RDD                        | DataFrame                    | Dataset                    |
|----------------|----------------------------|------------------------------|----------------------------|
| Abstraction    | Low-level                  | High-level                   | High-level                 |
| Schema         | No schema                  | Named columns with schema    | Typed schema               |
| Optimization   | No Catalyst                | Catalyst + Tungsten          | Catalyst + Tungsten        |
| Language       | Python, Scala, Java        | Python, Scala, Java, R       | Scala, Java only           |
| Type Safety    | No                         | No                           | Yes (compile-time)         |
| Use Case       | Fine-grained control       | SQL-like operations          | Type-safe + optimized      |

> **Note:** Dataset is NOT available in PySpark (Python). Use DataFrames instead.

```python
# RDD
rdd = sc.parallelize([(1, "Alice"), (2, "Bob")])

# DataFrame (preferred in PySpark)
df = spark.createDataFrame([(1, "Alice"), (2, "Bob")], ["id", "name"])
df.show()
```

---

#### Q4. What is lazy evaluation in Spark?

**Answer:**  
Lazy evaluation means **transformations are not executed immediately**. Spark builds a DAG (execution plan) and only executes when an action is called.

**Why it matters:**
- Allows Spark to optimize the entire pipeline before execution
- Avoids unnecessary computation
- Enables pipelining of operations

```python
# These lines do NOT execute immediately
df_filtered = df.filter(df["age"] > 25)          # Lazy
df_selected = df_filtered.select("name", "age")  # Lazy

# This triggers actual execution
df_selected.show()   # Action — executes NOW
df_selected.count()  # Action — executes NOW
```

---

#### Q5. Explain the Spark Architecture.

**Answer:**

```
┌─────────────────────────────────────────────────────┐
│                    Driver Program                    │
│  ┌──────────────┐    ┌──────────────────────────┐   │
│  │ SparkContext │    │   DAG Scheduler /        │   │
│  │              │───▶│   Task Scheduler         │   │
│  └──────────────┘    └──────────────────────────┘   │
└────────────────────────────┬────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │       Cluster Manager        │
              │  (YARN / Mesos / K8s /       │
              │   Standalone)                │
              └──────┬───────────┬───────────┘
                     │           │
          ┌──────────▼──┐   ┌────▼──────────┐
          │  Executor 1  │   │  Executor 2   │
          │  ┌────────┐  │   │  ┌────────┐   │
          │  │ Task 1 │  │   │  │ Task 3 │   │
          │  │ Task 2 │  │   │  │ Task 4 │   │
          │  └────────┘  │   │  └────────┘   │
          └──────────────┘   └───────────────┘
```

| Component       | Role                                                   |
|-----------------|--------------------------------------------------------|
| Driver          | Runs main(), creates SparkContext, manages job         |
| SparkContext    | Entry point, connects to cluster manager               |
| Cluster Manager | Allocates resources (YARN, Mesos, Kubernetes)          |
| Executor        | Runs tasks, stores cached data, reports to driver      |
| Task            | Smallest unit of work, runs on one partition           |

---

#### Q6. What is a DAG in Spark?

**Answer:**  
DAG (Directed Acyclic Graph) is Spark's **logical execution plan**. Each node represents an RDD/transformation and edges represent dependencies.

- **Stage** — Group of tasks that can run without a shuffle
- **Shuffle boundary** — Where one stage ends and another begins (e.g., `groupBy`, `join`)

```python
# DAG Example
rdd1 = sc.textFile("data.txt")           # Stage 1 starts
rdd2 = rdd1.flatMap(lambda x: x.split()) # Stage 1
rdd3 = rdd2.map(lambda x: (x, 1))        # Stage 1
rdd4 = rdd3.reduceByKey(lambda a, b: a+b) # Shuffle → Stage 2 starts
rdd4.collect()                             # Action — triggers DAG execution
```

---

#### Q7. What is the difference between `map()` and `flatMap()`?

**Answer:**

| Operation  | Input → Output   | Description                            |
|------------|------------------|----------------------------------------|
| `map()`    | 1 element → 1    | Applies function, keeps structure      |
| `flatMap()`| 1 element → many | Applies function, flattens the result  |

```python
rdd = sc.parallelize(["Hello World", "PySpark Interview"])

# map() — returns list of lists
rdd.map(lambda x: x.split()).collect()
# [['Hello', 'World'], ['PySpark', 'Interview']]

# flatMap() — flattens into single list
rdd.flatMap(lambda x: x.split()).collect()
# ['Hello', 'World', 'PySpark', 'Interview']
```

---

#### Q8. What is the difference between Transformations and Actions?

**Answer:**

| Property       | Transformations              | Actions                         |
|----------------|------------------------------|---------------------------------|
| Execution      | Lazy (not immediate)         | Eager (triggers execution)      |
| Returns        | New RDD / DataFrame          | Value / writes to storage       |
| Examples       | `filter`, `map`, `select`    | `collect`, `count`, `show`      |
| DAG effect     | Adds to DAG                  | Submits DAG for execution       |

```python
# Transformations — lazy
t1 = df.filter(df["salary"] > 50000)
t2 = t1.groupBy("dept").sum("salary")

# Actions — trigger execution
t2.show()          # ✅ Action
t2.count()         # ✅ Action
t2.collect()       # ✅ Action
t2.write.parquet() # ✅ Action
```

---

### 🔷 Section 2: DataFrames & Spark SQL (Q9–Q16)

---

#### Q9. How do you create a SparkSession?

```python
from pyspark.sql import SparkSession

# Basic
spark = SparkSession.builder \
    .appName("MyApp") \
    .getOrCreate()

# With configurations
spark = SparkSession.builder \
    .appName("MyApp") \
    .master("local[4]") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.executor.memory", "4g") \
    .enableHiveSupport() \
    .getOrCreate()

# Create DataFrame
df = spark.createDataFrame([
    (1, "Alice", 30),
    (2, "Bob",   25)
], ["id", "name", "age"])

df.printSchema()
df.show()
```

---

#### Q10. What is the difference between `select()` and `withColumn()`?

```python
from pyspark.sql import functions as F

# select() — choose/rename/transform specific columns
df.select("name", "age")
df.select(F.col("name"), (F.col("salary") * 1.1).alias("new_salary"))

# withColumn() — add or replace ONE column, keeps all others
df.withColumn("age_plus_10", F.col("age") + 10)
df.withColumn("name", F.upper(F.col("name")))  # Replace existing column

# Key difference:
# select() → returns only specified columns
# withColumn() → returns ALL columns + new/modified column
```

---

#### Q11. How do you handle null values in PySpark?

```python
from pyspark.sql import functions as F

# 1. Drop rows with ANY null
df.dropna()

# 2. Drop rows where specific columns are null
df.dropna(subset=["name", "salary"])

# 3. Fill nulls with static values
df.fillna({"age": 0, "name": "Unknown", "salary": 0.0})

# 4. Fill nulls using another column
df.withColumn("salary", F.coalesce(F.col("salary"), F.col("default_salary")))

# 5. Filter out nulls
df.filter(F.col("age").isNotNull())

# 6. Check for nulls
df.filter(F.col("name").isNull())

# 7. Replace nulls with mean
mean_val = df.select(F.mean("salary")).collect()[0][0]
df.fillna({"salary": mean_val})
```

---

#### Q12. What is the difference between `cache()` and `persist()`?

```python
from pyspark import StorageLevel

# cache() — shortcut for MEMORY_AND_DISK
df.cache()

# persist() — choose your storage level
df.persist(StorageLevel.MEMORY_ONLY)
df.persist(StorageLevel.MEMORY_AND_DISK)
df.persist(StorageLevel.DISK_ONLY)
df.persist(StorageLevel.MEMORY_ONLY_2)  # Replicated 2x

# Always unpersist when done
df.unpersist()

# When to use:
# cache()/persist() — DataFrames used multiple times in the same job
# Don't cache — DataFrames used only once (wastes memory)
```

| Storage Level      | Memory | Disk | Serialized | Replicated |
|--------------------|--------|------|------------|------------|
| MEMORY_ONLY        | ✅      | ❌    | ❌          | ❌          |
| MEMORY_AND_DISK    | ✅      | ✅    | ❌          | ❌          |
| DISK_ONLY          | ❌      | ✅    | ✅          | ❌          |
| MEMORY_ONLY_2      | ✅      | ❌    | ❌          | ✅          |

---

#### Q13. Explain `groupBy()` vs `partitionBy()`

```python
from pyspark.sql.window import Window
from pyspark.sql import functions as F

# groupBy() — SQL-style aggregation, collapses rows
df.groupBy("dept").agg(
    F.sum("salary").alias("total_salary"),
    F.avg("salary").alias("avg_salary"),
    F.count("*").alias("headcount")
)

# partitionBy() in Window — groups for window operations, keeps all rows
window = Window.partitionBy("dept").orderBy(F.desc("salary"))
df.withColumn("rank_in_dept", F.rank().over(window))

# partitionBy() in write — splits output files
df.write.partitionBy("year", "month").parquet("output/")
```

---

#### Q14. What are Window Functions? Give an example.

```python
from pyspark.sql.window import Window
from pyspark.sql import functions as F

# Define window
window = Window.partitionBy("dept").orderBy(F.desc("salary"))

df.withColumn("row_number", F.row_number().over(window)) \  # Unique rank, no ties
  .withColumn("rank",       F.rank().over(window)) \        # Gaps on ties
  .withColumn("dense_rank", F.dense_rank().over(window)) \  # No gaps on ties
  .withColumn("lag_salary", F.lag("salary", 1).over(window)) \    # Previous row
  .withColumn("lead_salary", F.lead("salary", 1).over(window)) \  # Next row
  .withColumn("running_sum", F.sum("salary").over(
      window.rowsBetween(Window.unboundedPreceding, Window.currentRow)
  ))
```

---

#### Q15. How do you perform joins in PySpark?

```python
# Basic joins
df1.join(df2, on="id", how="inner")
df1.join(df2, on="id", how="left")
df1.join(df2, on="id", how="right")
df1.join(df2, on="id", how="full")   # Full outer

# Semi join — keep left rows that match right (no right columns)
df1.join(df2, on="id", how="left_semi")

# Anti join — keep left rows that DON'T match right
df1.join(df2, on="id", how="left_anti")

# Cross join — cartesian product
df1.crossJoin(df2)

# Join on multiple columns
df1.join(df2, on=["dept_id", "location_id"], how="inner")

# Join with different column names
df1.join(df2, df1["emp_id"] == df2["employee_id"], how="left")

# Broadcast join (for small tables)
from pyspark.sql.functions import broadcast
df1.join(broadcast(df2), on="id")
```

---

#### Q16. What is the difference between `orderBy()` and `sortWithinPartitions()`?

```python
# orderBy() — Global sort (expensive, causes full shuffle)
df.orderBy(F.desc("salary"))
df.orderBy(["dept", "salary"], ascending=[True, False])

# sortWithinPartitions() — Local sort per partition (fast, no shuffle)
df.sortWithinPartitions(F.desc("salary"))

# When to use each:
# orderBy()              → Final output needs to be globally sorted
# sortWithinPartitions() → Pre-sort for downstream operations (joins, window)
#                          Writing sorted partitioned files
```

---

### 🔷 Section 3: Performance & Optimization (Q17–Q23)

---

#### Q17. What is a shuffle in Spark and why is it expensive?

**Answer:**  
A **shuffle** is the process of redistributing data across partitions, required by operations like `groupBy`, `join`, `distinct`, `repartition`.

**Why expensive:**
1. Data written to disk (spill)
2. Data transferred over the network
3. Data serialized and deserialized
4. Creates new stage boundary in DAG

```python
# Operations that cause shuffle:
df.groupBy("dept").count()          # Shuffle
df.join(df2, on="id")               # Shuffle (unless broadcast)
df.distinct()                       # Shuffle
df.repartition(100)                 # Shuffle
df.orderBy("salary")                # Shuffle

# Tune shuffle behavior
spark.conf.set("spark.sql.shuffle.partitions", "200")       # Default 200
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10mb")  # Auto-broadcast
```

---

#### Q18. What is broadcast join and when should you use it?

```python
from pyspark.sql.functions import broadcast

# Use when one table is small (< 10MB by default, configurable)
large_df.join(broadcast(small_df), on="product_id")

# Change threshold
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "50mb")  # 50MB

# Disable auto broadcast
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

# Verify broadcast in explain plan
large_df.join(broadcast(small_df), on="id").explain()
# Look for: BroadcastHashJoin in the plan
```

---

#### Q19. What is data skew and how do you handle it?

```python
# Detect skew — check partition sizes
df.groupBy(F.spark_partition_id()).count().show()

# Solution 1: Salting technique
import pyspark.sql.functions as F

SALT = 10

# Add random salt to large table
df_large = df_large.withColumn(
    "salted_key",
    F.concat(F.col("key"), F.lit("_"), (F.rand() * SALT).cast("int"))
)

# Explode small table to match all salts
df_small = df_small.withColumn(
    "salt_array", F.array([F.lit(i) for i in range(SALT)])
).withColumn("salt", F.explode("salt_array")) \
 .withColumn("salted_key", F.concat(F.col("key"), F.lit("_"), F.col("salt")))

result = df_large.join(df_small, on="salted_key").drop("salted_key", "salt")

# Solution 2: Broadcast small table
df_large.join(broadcast(df_small), on="key")

# Solution 3: Repartition on a different key
df.repartition(200, "non_skewed_col")
```

---

#### Q20. What is the difference between `repartition()` and `coalesce()`?

```python
# repartition() — FULL SHUFFLE, can increase OR decrease
df.repartition(200)                  # Set to 200 partitions
df.repartition(200, "city")          # Partition by column (hash)

# coalesce() — NO SHUFFLE, can only DECREASE
df.coalesce(10)                      # Reduce to 10 partitions (efficient)

# Check current partitions
print(df.rdd.getNumPartitions())

# Rule of thumb:
# Use repartition() → Need to increase partitions, or evenly distribute
# Use coalesce()    → Need to reduce partitions (e.g., before writing)

# Example: Writing one file per output folder
df.coalesce(1).write.parquet("output/")
```

---

#### Q21. What is the Catalyst Optimizer?

**Answer:**  
Catalyst is Spark SQL's **query optimizer**. It transforms user queries into efficient execution plans.

```
SQL / DataFrame API
        ↓
   Unresolved Logical Plan
        ↓  (Analysis — resolve column names, types)
   Resolved Logical Plan
        ↓  (Logical Optimization — predicate pushdown, constant folding)
   Optimized Logical Plan
        ↓  (Physical Planning — choose join strategy)
   Physical Plans  →  Cost Model → Best Physical Plan
        ↓  (Code Generation via Tungsten)
   Executed RDDs
```

```python
# See the optimizer at work
df.filter(F.col("age") > 25).select("name", "age").explain(True)

# Catalyst optimizations include:
# - Predicate pushdown (filter early)
# - Column pruning (read only needed columns)
# - Constant folding (evaluate constant expressions at compile time)
# - Join reordering
```

---

#### Q22. How do you use `explain()` for query optimization?

```python
# Basic explain
df.explain()

# Extended (logical + physical plan)
df.explain(True)

# Spark 3.0+ modes
df.explain("simple")     # Basic physical plan
df.explain("extended")   # Logical + physical
df.explain("codegen")    # Generated Java code
df.explain("cost")       # Cost-based plan
df.explain("formatted")  # Clean formatted output (best for reading)

# What to look for:
# ✅ BroadcastHashJoin   → Efficient small table join
# ⚠️ SortMergeJoin       → Large table join (shuffle involved)
# ✅ PushedFilters        → Filter pushed to data source
# ⚠️ Exchange            → Shuffle happening (expensive)
# ✅ InMemoryTableScan   → Reading from cache
```

---

#### Q23. What are Accumulators and Broadcast Variables?

```python
# ============ ACCUMULATORS ============
# Distributed counters — workers can only ADD, driver can READ

error_count = sc.accumulator(0)
null_count  = sc.accumulator(0)

def process_row(row):
    if row["status"] == "ERROR":
        error_count.add(1)
    if row["value"] is None:
        null_count.add(1)
    return row

df.rdd.foreach(process_row)

print(f"Errors: {error_count.value}")
print(f"Nulls:  {null_count.value}")


# ============ BROADCAST VARIABLES ============
# Read-only shared data cached on each executor — avoids repeated transfers

lookup_dict = {"NY": "New York", "CA": "California", "TX": "Texas"}
broadcast_lookup = sc.broadcast(lookup_dict)

def expand_state(row):
    return broadcast_lookup.value.get(row["state"], "Unknown")

# In DataFrame API — broadcast a DataFrame
large_df.join(broadcast(small_df), on="id")
```

---

### 🔷 Section 4: File Formats & I/O (Q24–Q27)

---

#### Q24. What file formats does PySpark support?

| Format   | Type       | Schema | Compression | Splittable | Best For                   |
|----------|------------|--------|-------------|------------|----------------------------|
| Parquet  | Columnar   | ✅      | ✅ (Snappy)  | ✅          | Analytics, default choice  |
| ORC      | Columnar   | ✅      | ✅           | ✅          | Hive workloads             |
| Avro     | Row        | ✅      | ✅           | ✅          | Streaming, Kafka           |
| Delta    | Columnar   | ✅      | ✅           | ✅          | ACID, upserts, time travel |
| CSV      | Row        | ❌      | ❌           | ✅          | Simple exchange            |
| JSON     | Row        | ❌      | ❌           | ✅          | APIs, semi-structured      |

> ✅ **Always prefer Parquet or Delta Lake in production**

---

#### Q25. How do you read and write data in PySpark?

```python
# ========== READ ==========
# Parquet
df = spark.read.parquet("s3://bucket/data/")

# CSV with options
df = spark.read.csv("data.csv",
    header=True,
    inferSchema=True,  # Avoid in production — use explicit schema
    sep=",",
    nullValue="NA",
    dateFormat="yyyy-MM-dd"
)

# JSON
df = spark.read.json("data.json")
df = spark.read.option("multiLine", True).json("nested.json")

# With explicit schema
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

schema = StructType([
    StructField("id",     IntegerType(), nullable=False),
    StructField("name",   StringType(),  nullable=True),
    StructField("salary", DoubleType(),  nullable=True)
])
df = spark.read.csv("data.csv", schema=schema, header=True)

# ========== WRITE ==========
# Parquet
df.write.mode("overwrite").parquet("output/")
df.write.mode("append").parquet("output/")

# Partitioned write
df.write \
  .mode("overwrite") \
  .partitionBy("year", "month") \
  .parquet("s3://bucket/output/")

# Control number of output files
df.coalesce(1).write.mode("overwrite").csv("single_file_output/")
df.repartition(10).write.parquet("ten_files_output/")
```

---

#### Q26. What is schema inference and why avoid it in production?

```python
# ❌ BAD — inferSchema scans entire file (doubles read time)
df = spark.read.csv("huge.csv", inferSchema=True, header=True)

# ✅ GOOD — Define schema explicitly
from pyspark.sql.types import *

schema = StructType([
    StructField("order_id",    LongType(),    False),
    StructField("customer_id", IntegerType(), True),
    StructField("amount",      DoubleType(),  True),
    StructField("order_date",  DateType(),    True),
    StructField("status",      StringType(),  True)
])

df = spark.read.csv("huge.csv", schema=schema, header=True)

# Benefits of explicit schema:
# ✅ Faster reads (no scanning pass)
# ✅ Correct data types guaranteed
# ✅ Fails fast on schema mismatch
# ✅ Self-documenting code
```

---

#### Q27. What is Delta Lake?

**Answer:**  
Delta Lake is an open-source storage layer that brings **ACID transactions** and **reliability** to data lakes.

```python
# Write as Delta
df.write.format("delta").mode("overwrite").save("/delta/sales")

# Read Delta
df = spark.read.format("delta").load("/delta/sales")

# UPSERT (MERGE)
from delta.tables import DeltaTable

delta_table = DeltaTable.forPath(spark, "/delta/sales")
delta_table.alias("target").merge(
    updates_df.alias("source"),
    "target.id = source.id"
).whenMatchedUpdateAll() \
 .whenNotMatchedInsertAll() \
 .execute()

# Time Travel — query historical data
df_v1 = spark.read.format("delta") \
    .option("versionAsOf", 1) \
    .load("/delta/sales")

df_yesterday = spark.read.format("delta") \
    .option("timestampAsOf", "2024-01-01") \
    .load("/delta/sales")

# View history
delta_table.history().show()
```

---

### 🔷 Section 5: Advanced Topics (Q28–Q30)

---

#### Q28. What is Structured Streaming in PySpark?

```python
# Read from Kafka stream
stream_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "orders_topic") \
    .load()

# Parse JSON payload
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

schema = StructType([
    StructField("order_id", StringType()),
    StructField("amount",   DoubleType())
])

parsed = stream_df \
    .selectExpr("CAST(value AS STRING) as json_str") \
    .select(F.from_json("json_str", schema).alias("data")) \
    .select("data.*")

# Aggregation on stream
agg = parsed.groupBy(
    F.window("timestamp", "5 minutes")
).agg(F.sum("amount").alias("total_revenue"))

# Write stream output
query = agg.writeStream \
    .outputMode("update") \
    .format("console") \
    .trigger(processingTime="10 seconds") \
    .start()

query.awaitTermination()
```

---

#### Q29. How does fault tolerance work in Spark?

**Answer:**

| Mechanism          | How It Works                                           |
|--------------------|--------------------------------------------------------|
| RDD Lineage        | Recomputes lost partitions from parent RDDs via DAG    |
| Checkpointing      | Saves RDD to HDFS, truncates long lineage chains       |
| WAL                | Write-Ahead Logs for streaming — exactly-once delivery |
| Replication        | MEMORY_ONLY_2 stores partitions on 2 nodes             |

```python
# Enable checkpointing
sc.setCheckpointDir("hdfs:///checkpoints/")

rdd = sc.parallelize(range(1000))
# After a long chain of transformations:
rdd_transformed.checkpoint()

# For Structured Streaming
query = df.writeStream \
    .option("checkpointLocation", "hdfs:///streaming/checkpoints/") \
    .format("parquet") \
    .start("output/")
```

---

#### Q30. What is the difference between Client Mode and Cluster Mode?

| Feature             | Client Mode                     | Cluster Mode                    |
|---------------------|---------------------------------|---------------------------------|
| Driver location     | Local machine (submitting node) | One of the worker/cluster nodes |
| Best for            | Interactive, debugging          | Production jobs                 |
| Network             | High dependency on local machine| Low — all inside cluster        |
| Logs                | Visible locally                 | Must fetch from cluster         |
| If client dies      | Job fails                       | Job continues                   |

```bash
# Client mode (default for spark-submit)
spark-submit \
  --master yarn \
  --deploy-mode client \
  my_job.py

# Cluster mode (recommended for production)
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --num-executors 10 \
  --executor-memory 4g \
  --executor-cores 2 \
  my_job.py
```

---

## 💻 PART 2 — Top 30 Scenario-Based Coding Questions

---

### 🗂️ Section 1: Data Cleaning & Transformation (Q1–Q8)

---

#### Q1. Remove duplicates based on specific columns and keep the latest record.

**Scenario:** You have a customer table with duplicate `customer_id` rows. Keep only the most recent record per customer.

```python
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Sample data
data = [
    (1, "Alice", "2024-01-15"),
    (1, "Alice", "2024-03-20"),  # Latest — keep this
    (2, "Bob",   "2024-02-10"),
    (2, "Bob",   "2024-01-05"),  # Older — drop this
]
df = spark.createDataFrame(data, ["customer_id", "name", "created_date"])

# Solution
window = Window.partitionBy("customer_id").orderBy(F.desc("created_date"))

result = df \
    .withColumn("rank", F.row_number().over(window)) \
    .filter(F.col("rank") == 1) \
    .drop("rank")

result.show()
# Output: One row per customer — the most recent one
```

---

#### Q2. Flatten a nested JSON / struct column.

**Scenario:** Your data has nested JSON with address as a struct. Flatten it to top-level columns.

```python
from pyspark.sql.types import StructType, StructField, StringType

# Sample nested data
data = [(1, ("New York", "10001", "NY")),
        (2, ("Los Angeles", "90001", "CA"))]

schema = StructType([
    StructField("id", StringType()),
    StructField("address", StructType([
        StructField("city",  StringType()),
        StructField("zip",   StringType()),
        StructField("state", StringType())
    ]))
])
df = spark.createDataFrame(data, schema)

# Method 1: Manual flatten
df.select(
    "id",
    "address.city",
    "address.zip",
    "address.state"
)

# Method 2: Dynamic flatten (handles any depth)
def flatten_df(df):
    from pyspark.sql.types import StructType
    cols = []
    for field in df.schema.fields:
        if isinstance(field.dataType, StructType):
            for subfield in field.dataType.fields:
                cols.append(
                    F.col(f"{field.name}.{subfield.name}")
                     .alias(f"{field.name}_{subfield.name}")
                )
        else:
            cols.append(F.col(field.name))
    return df.select(cols)

flat_df = flatten_df(df)
flat_df.show()

# Method 3: Explode array column
df.withColumn("item", F.explode("items_array"))
```

---

#### Q3. Pivot a table — convert row values into columns.

**Scenario:** Sales data has (year, product, amount) rows. Pivot to show products as columns.

```python
# Sample data
data = [
    (2023, "TV",     5000),
    (2023, "Phone",  3000),
    (2023, "Laptop", 7000),
    (2024, "TV",     5500),
    (2024, "Phone",  3200),
    (2024, "Laptop", 8000),
]
df = spark.createDataFrame(data, ["year", "product", "amount"])

# Pivot
result = df.groupBy("year") \
    .pivot("product", ["TV", "Phone", "Laptop"]) \
    .agg(F.sum("amount"))

result.show()
# +----+----+-----+------+
# |year|  TV|Phone|Laptop|
# +----+----+-----+------+
# |2023|5000| 3000|  7000|
# |2024|5500| 3200|  8000|
# +----+----+-----+------+
```

---

#### Q4. Unpivot (melt) columns back into rows.

**Scenario:** You have a wide table with product columns. Convert back to long format.

```python
# Wide format
data = [(2023, 5000, 3000, 7000),
        (2024, 5500, 3200, 8000)]
df = spark.createDataFrame(data, ["year", "TV", "Phone", "Laptop"])

# Unpivot using stack()
result = df.select(
    "year",
    F.expr("stack(3, 'TV', TV, 'Phone', Phone, 'Laptop', Laptop) as (product, amount)")
)

result.show()
# +----+-------+------+
# |year|product|amount|
# +----+-------+------+
# |2023|     TV|  5000|
# |2023|  Phone|  3000|
# |2023| Laptop|  7000|
# |2024|     TV|  5500|
# ...
```

---

#### Q5. Replace nulls with the previous non-null value (forward fill).

**Scenario:** Time series data has missing values. Fill each null with the last known value.

```python
data = [
    (1, "2024-01-01", 100.0),
    (1, "2024-01-02", None),
    (1, "2024-01-03", None),
    (1, "2024-01-04", 150.0),
    (2, "2024-01-01", None),
    (2, "2024-01-02", 200.0),
]
df = spark.createDataFrame(data, ["id", "date", "value"])

# Forward fill
window = Window.partitionBy("id") \
               .orderBy("date") \
               .rowsBetween(Window.unboundedPreceding, Window.currentRow)

result = df.withColumn(
    "value_filled",
    F.last("value", ignorenulls=True).over(window)
)

result.show()
# value_filled uses last known value for nulls
```

---

#### Q6. Standardize phone numbers — keep only digits, enforce 10-digit format.

**Scenario:** Phone numbers arrive in various formats. Clean and validate them.

```python
data = [
    (1, "+1-800-555-1234"),
    (2, "(415) 555-2671"),
    (3, "555.867.5309"),
    (4, "12345"),            # Invalid — too short
    (5, "8005551234"),       # Already clean
]
df = spark.createDataFrame(data, ["id", "phone"])

result = df \
    .withColumn("digits_only",
        F.regexp_replace(F.col("phone"), r"[^0-9]", "")
    ) \
    .withColumn("phone_10",
        F.when(F.length("digits_only") == 11,
            F.substring("digits_only", 2, 10)  # Remove leading country code 1
        ).otherwise(F.col("digits_only"))
    ) \
    .withColumn("phone_valid",
        F.when(F.length("phone_10") == 10, F.col("phone_10"))
         .otherwise(F.lit(None))
    )

result.select("id", "phone", "phone_valid").show()
```

---

#### Q7. Parse and split a full name column into first and last name.

```python
data = [(1, "Alice Johnson"), (2, "Bob Smith Jr"), (3, "Madonna")]
df = spark.createDataFrame(data, ["id", "full_name"])

result = df \
    .withColumn("name_parts", F.split(F.col("full_name"), " ")) \
    .withColumn("first_name", F.col("name_parts")[0]) \
    .withColumn("last_name",
        F.when(F.size("name_parts") > 1,
            F.col("name_parts")[F.size("name_parts") - 1]
        ).otherwise(F.lit(None))
    ) \
    .drop("name_parts")

result.show()
```

---

#### Q8. Convert multiple date formats in one column to a standard format.

**Scenario:** Date column has mixed formats from different source systems.

```python
data = [
    (1, "2024-03-15"),
    (2, "03/15/2024"),
    (3, "15-Mar-2024"),
    (4, "20240315"),
    (5, "bad_date"),   # Invalid
]
df = spark.createDataFrame(data, ["id", "raw_date"])

result = df.withColumn("std_date",
    F.coalesce(
        F.to_date("raw_date", "yyyy-MM-dd"),
        F.to_date("raw_date", "MM/dd/yyyy"),
        F.to_date("raw_date", "dd-MMM-yyyy"),
        F.to_date("raw_date", "yyyyMMdd")
    )
).withColumn("is_valid", F.col("std_date").isNotNull())

result.show()
```

---

### 📊 Section 2: Aggregations & Window Functions (Q9–Q15)

---

#### Q9. Calculate running total (cumulative sum) per customer.

```python
data = [
    (1, "2024-01-01", 100),
    (1, "2024-01-05", 200),
    (1, "2024-01-10", 150),
    (2, "2024-01-02", 300),
    (2, "2024-01-08", 100),
]
df = spark.createDataFrame(data, ["customer_id", "order_date", "amount"])

window = Window.partitionBy("customer_id") \
               .orderBy("order_date") \
               .rowsBetween(Window.unboundedPreceding, Window.currentRow)

result = df.withColumn("running_total", F.sum("amount").over(window))
result.show()
# Each row shows cumulative spend for that customer up to that date
```

---

#### Q10. Find the top N products per category by revenue.

```python
data = [
    ("Electronics", "TV",     5000),
    ("Electronics", "Phone",  3000),
    ("Electronics", "Laptop", 7000),
    ("Electronics", "Tablet", 2000),
    ("Clothing",    "Jacket", 1500),
    ("Clothing",    "Shoes",  2500),
    ("Clothing",    "Shirt",   500),
]
df = spark.createDataFrame(data, ["category", "product", "revenue"])

N = 2  # Top 2 per category
window = Window.partitionBy("category").orderBy(F.desc("revenue"))

result = df \
    .withColumn("rank", F.dense_rank().over(window)) \
    .filter(F.col("rank") <= N) \
    .drop("rank")

result.show()
```

---

#### Q11. Calculate month-over-month (MoM) revenue growth %.

```python
data = [
    ("US", 2024, 1, 10000),
    ("US", 2024, 2, 12000),
    ("US", 2024, 3, 11000),
    ("US", 2024, 4, 15000),
]
df = spark.createDataFrame(data, ["region", "year", "month", "revenue"])

window = Window.partitionBy("region").orderBy("year", "month")

result = df \
    .withColumn("prev_revenue", F.lag("revenue", 1).over(window)) \
    .withColumn("mom_growth_pct",
        F.when(F.col("prev_revenue").isNotNull(),
            F.round(
                (F.col("revenue") - F.col("prev_revenue")) / F.col("prev_revenue") * 100,
                2
            )
        ).otherwise(F.lit(None))
    )

result.show()
```

---

#### Q12. Find customers who placed orders in consecutive months.

```python
data = [
    (1, "2024-01-01"),
    (1, "2024-02-15"),  # Consecutive with Jan
    (1, "2024-04-10"),  # Gap — not consecutive with Feb
    (2, "2024-01-01"),
    (2, "2024-02-01"),
    (2, "2024-03-01"),  # All 3 months consecutive
]
df = spark.createDataFrame(data, ["customer_id", "order_date"])
df = df.withColumn("order_date", F.to_date("order_date")) \
       .withColumn("order_month", F.date_trunc("month", "order_date"))

window = Window.partitionBy("customer_id").orderBy("order_month")

result = df \
    .withColumn("prev_month", F.lag("order_month", 1).over(window)) \
    .withColumn("months_gap",
        F.months_between(F.col("order_month"), F.col("prev_month"))
    ) \
    .withColumn("is_consecutive",
        F.when(F.col("months_gap") == 1, True).otherwise(False)
    ) \
    .filter(F.col("is_consecutive") == True)

result.show()
```

---

#### Q13. Calculate 7-day rolling average of sales.

```python
data = [
    ("store_1", "2024-01-01", 1000),
    ("store_1", "2024-01-02", 1200),
    ("store_1", "2024-01-03", 900),
    ("store_1", "2024-01-07", 1100),
    ("store_1", "2024-01-08", 1300),
]
df = spark.createDataFrame(data, ["store_id", "sale_date", "sales"])
df = df.withColumn("sale_date", F.to_date("sale_date")) \
       .withColumn("date_long", F.col("sale_date").cast("long"))

SECONDS_PER_DAY = 86400

window = Window.partitionBy("store_id") \
               .orderBy("date_long") \
               .rangeBetween(-6 * SECONDS_PER_DAY, 0)

result = df.withColumn("rolling_7d_avg",
    F.round(F.avg("sales").over(window), 2)
)

result.show()
```

---

#### Q14. Assign percentile buckets (quartiles) to customers by spend.

```python
data = [(i, i * 100) for i in range(1, 21)]
df = spark.createDataFrame(data, ["customer_id", "total_spend"])

window = Window.orderBy("total_spend")

result = df \
    .withColumn("quartile",  F.ntile(4).over(window)) \
    .withColumn("decile",    F.ntile(10).over(window)) \
    .withColumn("percentile",
        F.percent_rank().over(window)
    )

result.show()
# quartile 1 = bottom 25%, 4 = top 25%
```

---

#### Q15. Find the second highest salary in each department.

```python
data = [
    ("Engineering", "Alice", 120000),
    ("Engineering", "Bob",   95000),
    ("Engineering", "Carol", 110000),
    ("Marketing",   "Dave",  80000),
    ("Marketing",   "Eve",   85000),
]
df = spark.createDataFrame(data, ["dept", "employee", "salary"])

window = Window.partitionBy("dept").orderBy(F.desc("salary"))

result = df \
    .withColumn("dense_rank", F.dense_rank().over(window)) \
    .filter(F.col("dense_rank") == 2) \
    .select("dept", "employee", "salary")

result.show()
```

---

### 🔗 Section 3: Joins & Set Operations (Q16–Q20)

---

#### Q16. Find customers who exist in table A but NOT in table B (anti join).

**Scenario:** Find customers who registered but never placed an order.

```python
customers = spark.createDataFrame([
    (1, "Alice"), (2, "Bob"), (3, "Carol"), (4, "Dave")
], ["customer_id", "name"])

orders = spark.createDataFrame([
    (1, "ORD001"), (3, "ORD002")
], ["customer_id", "order_id"])

# Customers with NO orders
no_orders = customers.join(orders, on="customer_id", how="left_anti")
no_orders.show()
# Returns Bob and Dave
```

---

#### Q17. Deduplicate across two DataFrames and combine (union distinct).

```python
df_source1 = spark.createDataFrame([
    (1, "Alice", 30), (2, "Bob", 25)
], ["id", "name", "age"])

df_source2 = spark.createDataFrame([
    (2, "Bob", 25),   # Duplicate
    (3, "Carol", 35)
], ["id", "name", "age"])

# Union and remove duplicates
result = df_source1.unionByName(df_source2).dropDuplicates()
result.show()

# Deduplicate on specific columns only
result = df_source1.unionByName(df_source2).dropDuplicates(["id"])
result.show()
```

---

#### Q18. Perform a fuzzy/approximate join on names using similarity.

**Scenario:** Two systems have customer names with slight variations. Match them.

```python
df1 = spark.createDataFrame([
    (1, "John Smith"),
    (2, "Alice Johnson")
], ["id1", "name1"])

df2 = spark.createDataFrame([
    (101, "Jon Smith"),    # Typo — should match John Smith
    (102, "Alice Johnston")
], ["id2", "name2"])

# Using Levenshtein distance (max 3 edit distance)
result = df1.crossJoin(df2) \
    .withColumn("distance", F.levenshtein(F.col("name1"), F.col("name2"))) \
    .filter(F.col("distance") <= 3) \
    .orderBy("distance")

result.show()
```

---

#### Q19. Slowly Changing Dimension Type 2 — detect new/changed records.

**Scenario:** Daily snapshot arrives. Detect what changed since yesterday for SCD Type 2.

```python
current = spark.createDataFrame([
    (1, "Alice", "NY",  "2024-01-01"),
    (2, "Bob",   "CA",  "2024-01-01"),
], ["id", "name", "city", "effective_date"])

new_snapshot = spark.createDataFrame([
    (1, "Alice", "TX"),   # City changed NY → TX
    (2, "Bob",   "CA"),   # No change
    (3, "Carol", "FL"),   # New record
], ["id", "name", "city"])

# Find changed or new records
changed = new_snapshot.alias("new") \
    .join(current.alias("curr"), on="id", how="left") \
    .filter(
        F.col("curr.id").isNull() |              # New record
        (F.col("curr.city") != F.col("new.city"))  # Changed record
    ) \
    .select(
        F.col("new.id"),
        F.col("new.name"),
        F.col("new.city"),
        F.current_date().alias("effective_date")
    )

changed.show()
```

---

#### Q20. Join two large tables efficiently to avoid OOM (skewed join).

**Scenario:** One key (`customer_id = 0` for guest) dominates 80% of rows.

```python
import pyspark.sql.functions as F

# Simulate skewed data — customer_id=0 is 80% of rows
large_orders = spark.range(1000000).withColumn(
    "customer_id",
    F.when(F.rand() < 0.8, F.lit(0)).otherwise((F.rand() * 100).cast("int"))
).withColumn("amount", (F.rand() * 1000).cast("int"))

customers = spark.createDataFrame(
    [(i, f"Customer_{i}") for i in range(101)],
    ["customer_id", "customer_name"]
)

# Solution: Salt the skewed key
SALT = 10

# Salt the large table — random salt
orders_salted = large_orders.withColumn(
    "salted_key",
    F.concat(F.col("customer_id"), F.lit("_"), (F.rand() * SALT).cast("int"))
)

# Explode the small table for all salt values
from pyspark.sql.functions import array, explode, lit

customers_exploded = customers \
    .withColumn("salt_arr", F.array([F.lit(i) for i in range(SALT)])) \
    .withColumn("salt", F.explode("salt_arr")) \
    .withColumn("salted_key", F.concat(F.col("customer_id"), F.lit("_"), F.col("salt"))) \
    .drop("salt_arr", "salt")

# Join on salted key
result = orders_salted.join(customers_exploded, on="salted_key") \
    .drop("salted_key", "customer_id")

result.show(5)
```

---

### ⚡ Section 4: Performance & Optimization (Q21–Q25)

---

#### Q21. Your Spark job is very slow — how do you diagnose and fix it?

```python
# ============ DIAGNOSIS ============

# Step 1: Check query plan for expensive operations
df.explain("formatted")
# Look for: Exchange (shuffle), SortMergeJoin, CartesianProduct

# Step 2: Check partition count and skew
print("Partition count:", df.rdd.getNumPartitions())
df.groupBy(F.spark_partition_id().alias("partition")) \
  .count() \
  .orderBy(F.desc("count")) \
  .show(20)

# Step 3: Check Spark UI at http://localhost:4040
# Look for: long stages, high GC time, spill to disk

# ============ FIXES ============

# Fix 1: Too many small partitions → increase
df = df.repartition(200)

# Fix 2: Too few partitions → reduce (use coalesce)
df = df.coalesce(10)

# Fix 3: Skewed join → broadcast small table
df_result = large_df.join(broadcast(small_df), on="id")

# Fix 4: Reuse DataFrame → cache it
df.cache()

# Fix 5: Tune shuffle partitions
spark.conf.set("spark.sql.shuffle.partitions", "400")

# Fix 6: Use adaptive query execution (Spark 3+)
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
```

---

#### Q22. Read a 500GB CSV efficiently without OOM.

```python
from pyspark.sql.types import *

# ✅ Define schema explicitly — never inferSchema for large files
schema = StructType([
    StructField("order_id",    LongType(),    False),
    StructField("customer_id", IntegerType(), True),
    StructField("product_id",  IntegerType(), True),
    StructField("amount",      DoubleType(),  True),
    StructField("order_date",  StringType(),  True),
    StructField("status",      StringType(),  True)
])

# ✅ Read with optimized options
df = spark.read.csv(
    "s3://bucket/large_orders/*.csv",
    schema=schema,
    header=True,
    mode="DROPMALFORMED"  # Skip bad rows
)

# ✅ Select only needed columns immediately
df = df.select("order_id", "customer_id", "amount", "order_date")

# ✅ Filter early to reduce data volume
df = df.filter(F.col("order_date") >= "2024-01-01")

# ✅ Repartition for parallelism (~200MB per partition target)
# 500GB / 200MB = 2500 partitions
df = df.repartition(2500)

# ✅ Convert to Parquet for future runs
df.write.mode("overwrite").parquet("s3://bucket/orders_parquet/")
```

---

#### Q23. Write output partitioned by date and region efficiently.

```python
# ✅ Add partition columns if not present
df = df \
    .withColumn("year",  F.year("order_date")) \
    .withColumn("month", F.month("order_date")) \
    .withColumn("region", F.col("country_code"))

# ✅ Repartition by partition keys before writing (avoids too many small files)
df = df.repartition("year", "month", "region")

# ✅ Write partitioned output
df.write \
  .mode("overwrite") \
  .partitionBy("year", "month", "region") \
  .parquet("s3://bucket/sales_partitioned/")

# ✅ Control files per partition (1 file per partition folder)
df.repartition(F.col("year"), F.col("month"), F.col("region")) \
  .write \
  .mode("overwrite") \
  .partitionBy("year", "month", "region") \
  .parquet("s3://bucket/sales/")

# ✅ Avoid too many small files — check output
# Target: 128MB–512MB per file
```

---

#### Q24. Avoid reading entire table — use predicate pushdown.

```python
# ✅ Predicate pushdown — Spark pushes filters to data source
df = spark.read.parquet("s3://bucket/sales/") \
    .filter(F.col("year") == 2024) \       # Pushed to Parquet reader
    .filter(F.col("region") == "US") \     # Pushed to Parquet reader
    .filter(F.col("amount") > 1000)        # Pushed to Parquet reader

# Verify pushdown happened
df.explain()
# Look for: PushedFilters: [IsNotNull(year), EqualTo(year,2024), ...]

# ✅ Use partition pruning — reads only matching folders
df = spark.read.parquet("s3://bucket/sales/year=2024/region=US/")

# ✅ JDBC predicate pushdown
df = spark.read.jdbc(
    url="jdbc:postgresql://host/db",
    table="orders",
    properties={"user": "user", "password": "pass"},
    predicates=["order_date >= '2024-01-01'", "region = 'US'"]
)
```

---

#### Q25. Process 1 billion rows with limited cluster resources.

```python
# Strategy: Filter → Project → Aggregate → Join (in that order)

# ✅ Step 1: Read only needed columns (column pruning)
df = spark.read.parquet("s3://bucket/events/") \
    .select("user_id", "event_type", "amount", "event_date")

# ✅ Step 2: Filter early — reduce row count ASAP
df = df.filter(
    (F.col("event_date") >= "2024-01-01") &
    (F.col("event_type").isin(["purchase", "refund"]))
)

# ✅ Step 3: Aggregate before joining (reduce to summary)
summary = df.groupBy("user_id").agg(
    F.sum("amount").alias("total_spend"),
    F.count("*").alias("event_count")
)

# ✅ Step 4: Broadcast small dimension tables
users = spark.read.parquet("s3://bucket/users/")  # Small table
result = summary.join(broadcast(users), on="user_id")

# ✅ Step 5: Tune for large jobs
spark.conf.set("spark.sql.shuffle.partitions", "2000")
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.memory.fraction", "0.8")

# ✅ Step 6: Write incrementally
result.write \
    .mode("overwrite") \
    .partitionBy("event_date") \
    .parquet("s3://output/")
```

---

### 🧠 Section 5: Real-World Business Scenarios (Q26–Q30)

---

#### Q26. Identify churned customers — no purchase in last 90 days.

```python
from pyspark.sql.functions import current_date, datediff, max as spark_max

# Sample orders
data = [
    (1, "2024-01-15"),
    (1, "2024-02-20"),
    (2, "2023-10-01"),   # Churned — last purchase > 90 days ago
    (2, "2023-11-01"),
    (3, "2024-03-10"),   # Active
]
df = spark.createDataFrame(data, ["customer_id", "purchase_date"])
df = df.withColumn("purchase_date", F.to_date("purchase_date"))

# Find last purchase per customer
last_purchase = df.groupBy("customer_id") \
    .agg(spark_max("purchase_date").alias("last_purchase_date"))

# Classify churn
result = last_purchase \
    .withColumn("days_inactive",
        datediff(current_date(), F.col("last_purchase_date"))
    ) \
    .withColumn("status",
        F.when(F.col("days_inactive") > 90, "Churned")
         .when(F.col("days_inactive") > 30, "At Risk")
         .otherwise("Active")
    )

result.show()
```

---

#### Q27. Build a session window — group user events within 30-min gaps.

**Scenario:** Assign a session ID to each user's events, where a new session starts after 30 minutes of inactivity.

```python
data = [
    (1, "2024-01-01 10:00:00"),
    (1, "2024-01-01 10:05:00"),  # Same session (5 min gap)
    (1, "2024-01-01 10:45:00"),  # New session (40 min gap > 30)
    (1, "2024-01-01 10:50:00"),  # Same session as above
    (2, "2024-01-01 09:00:00"),
    (2, "2024-01-01 09:40:00"),  # New session
]
df = spark.createDataFrame(data, ["user_id", "event_time"])
df = df.withColumn("event_time", F.to_timestamp("event_time"))

window = Window.partitionBy("user_id").orderBy("event_time")

result = df \
    .withColumn("prev_time", F.lag("event_time", 1).over(window)) \
    .withColumn("gap_seconds",
        F.unix_timestamp("event_time") - F.unix_timestamp("prev_time")
    ) \
    .withColumn("new_session_flag",
        F.when(
            F.col("gap_seconds").isNull() |  # First event
            (F.col("gap_seconds") > 1800),   # Gap > 30 min
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

result.show(truncate=False)
```

---

#### Q28. Detect anomalies — flag transactions 3 standard deviations above mean.

**Scenario:** Flag suspicious transactions per category using Z-score.

```python
data = [
    ("Electronics", 500),
    ("Electronics", 520),
    ("Electronics", 480),
    ("Electronics", 9999),  # Anomaly!
    ("Clothing",    100),
    ("Clothing",    110),
    ("Clothing",    95),
    ("Clothing",    950),   # Anomaly!
]
df = spark.createDataFrame(data, ["category", "amount"])
df = df.withColumn("txn_id", F.monotonically_increasing_id())

# Calculate mean and stddev per category
stats = df.groupBy("category").agg(
    F.mean("amount").alias("mean_amount"),
    F.stddev("amount").alias("stddev_amount")
)

# Join stats and compute Z-score
result = df.join(stats, on="category") \
    .withColumn("z_score",
        F.abs(F.col("amount") - F.col("mean_amount")) / F.col("stddev_amount")
    ) \
    .withColumn("is_anomaly", F.col("z_score") > 3) \
    .orderBy(F.desc("z_score"))

result.select("category", "amount", F.round("z_score", 2).alias("z_score"), "is_anomaly").show()
```

---

#### Q29. Reconcile two data sources — find mismatches row by row.

**Scenario:** Compare source and target tables after an ETL migration.

```python
source = spark.createDataFrame([
    (1, "Alice", 1000.0),
    (2, "Bob",   2000.0),
    (3, "Carol", 1500.0),
    (4, "Dave",  3000.0),  # Extra in source
], ["id", "name", "amount"])

target = spark.createDataFrame([
    (1, "Alice", 1000.0),
    (2, "Bob",   2500.0),  # Amount mismatch
    (3, "Carol", 1500.0),
    (5, "Eve",   1200.0),  # Extra in target
], ["id", "name", "amount"])

# 1. Missing from target
missing_in_target = source.join(target, on="id", how="left_anti")
print("Missing in target:")
missing_in_target.show()

# 2. Extra in target (not in source)
extra_in_target = target.join(source, on="id", how="left_anti")
print("Extra in target:")
extra_in_target.show()

# 3. Value mismatches
mismatch = source.alias("s").join(target.alias("t"), on="id") \
    .filter(
        (F.col("s.amount") != F.col("t.amount")) |
        (F.col("s.name") != F.col("t.name"))
    ) \
    .select(
        "id",
        F.col("s.name").alias("source_name"),
        F.col("t.name").alias("target_name"),
        F.col("s.amount").alias("source_amount"),
        F.col("t.amount").alias("target_amount")
    )
print("Value mismatches:")
mismatch.show()

# 4. Summary report
print(f"Total source rows:     {source.count()}")
print(f"Total target rows:     {target.count()}")
print(f"Missing in target:     {missing_in_target.count()}")
print(f"Extra in target:       {extra_in_target.count()}")
print(f"Value mismatches:      {mismatch.count()}")
```

---

#### Q30. Generate a daily report — active users, revenue, top product per day.

```python
data = [
    ("2024-01-01", "user1", "TV",     5000),
    ("2024-01-01", "user2", "Phone",  3000),
    ("2024-01-01", "user1", "Laptop", 7000),
    ("2024-01-02", "user3", "TV",     4500),
    ("2024-01-02", "user4", "Phone",  3200),
    ("2024-01-02", "user3", "Laptop", 8000),
]
df = spark.createDataFrame(data, ["sale_date", "user_id", "product", "amount"])

# Daily revenue and active users per product
daily_product = df.groupBy("sale_date", "product").agg(
    F.countDistinct("user_id").alias("active_users"),
    F.sum("amount").alias("daily_revenue")
)

# Rank products per day
window = Window.partitionBy("sale_date").orderBy(F.desc("daily_revenue"))

daily_report = daily_product \
    .withColumn("rank", F.row_number().over(window)) \
    .withColumn("is_top_product", F.col("rank") == 1) \
    .orderBy("sale_date", "rank")

# Overall daily summary
daily_summary = df.groupBy("sale_date").agg(
    F.countDistinct("user_id").alias("total_active_users"),
    F.sum("amount").alias("total_revenue"),
    F.count("*").alias("total_transactions")
)

print("=== Daily Product Report ===")
daily_report.show()

print("=== Daily Summary ===")
daily_summary.orderBy("sale_date").show()
```

---

## 📌 Quick Reference Cheat Sheet

### Common PySpark Functions

```python
# String functions
F.upper(), F.lower(), F.trim(), F.ltrim(), F.rtrim()
F.substring(col, pos, len)
F.concat(col1, F.lit("-"), col2)
F.concat_ws("-", col1, col2)
F.regexp_replace(col, pattern, replacement)
F.regexp_extract(col, pattern, group)
F.split(col, delimiter)
F.length(col)

# Date functions
F.current_date(), F.current_timestamp()
F.to_date(col, format), F.to_timestamp(col, format)
F.date_add(col, days), F.date_sub(col, days)
F.datediff(end, start)
F.months_between(end, start)
F.year(col), F.month(col), F.dayofmonth(col)
F.date_trunc("month", col)
F.date_format(col, "yyyy-MM-dd")

# Conditional functions
F.when(condition, value).when(...).otherwise(default)
F.coalesce(col1, col2, col3)          # First non-null
F.nullif(col, value)                   # Returns null if equal
F.isnull(col), F.isnotnull(col)

# Aggregate functions
F.count(), F.countDistinct()
F.sum(), F.avg(), F.mean()
F.min(), F.max()
F.first(), F.last()
F.collect_list(), F.collect_set()
F.stddev(), F.variance()
F.percentile_approx(col, 0.5)

# Array functions
F.array(*cols)
F.explode(col)
F.array_contains(col, value)
F.array_distinct(col)
F.array_union(col1, col2)
F.size(col)

# Window functions
F.row_number(), F.rank(), F.dense_rank()
F.lag(col, n), F.lead(col, n)
F.first(), F.last()
F.sum(), F.avg(), F.min(), F.max()
F.ntile(n)
F.percent_rank(), F.cume_dist()
```

---

### Performance Tuning Configuration Reference

```python
# Shuffle
spark.conf.set("spark.sql.shuffle.partitions", "200")           # Default 200

# Broadcast
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10mb")  # Default 10MB

# Adaptive Query Execution (Spark 3.0+)
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

# Memory
spark.conf.set("spark.memory.fraction", "0.8")
spark.conf.set("spark.memory.storageFraction", "0.5")

# Serialization (use Kryo for performance)
spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

# Dynamic allocation
spark.conf.set("spark.dynamicAllocation.enabled", "true")
spark.conf.set("spark.dynamicAllocation.minExecutors", "2")
spark.conf.set("spark.dynamicAllocation.maxExecutors", "20")
```

---

*Happy Coding! 🚀 Push this to your repo and use GitHub Copilot to extend each example with sample data and unit tests.*
