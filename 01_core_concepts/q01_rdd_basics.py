"""
Q1: RDD Basics — Create, transform, and count RDDs

Scenario: Learn how to create RDDs, apply transformations, and trigger actions.
RDDs are the fundamental data structure of Apache Spark — Resilient Distributed Datasets.

Key Concepts:
- RDD (Resilient Distributed Dataset): Low-level, immutable, distributed collection
- Transformations: Lazy operations (map, filter, flatMap, union, etc.)
- Actions: Eager operations that trigger execution (collect, count, saveAsTextFile, etc.)
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - parallelize(): O(n) — distributes data across partitions
#   - map()/filter(): O(n) — operates on each element once
#   - distinct(): O(n*log(n)) — requires shuffle
#   - reduceByKey(): O(n) in executor, O(k) for reduce (k = unique keys)
#   - collect(): O(n) — transfers all data to driver (EXPENSIVE!)
#
# Shuffle Operations:
#   - distinct() — FULL SHUFFLE (all partitions exchange data)
#   - reduceByKey() — FULL SHUFFLE (redistribute by key)
#   - groupByKey() — FULL SHUFFLE (redistribute by key)
#   - union() — NO SHUFFLE (concatenates RDDs)
#
# Performance Tips:
#   - Avoid collect() on large RDDs (transfers to driver memory)
#   - Use reduceByKey() instead of groupByKey() when aggregating
#   - Cache RDDs that are reused multiple times
#   - Prefer DataFrame API (better optimization via Catalyst)
#   - Use appropriate partition count (2-4x num cores)
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Q01_RDD_Basics") \
    .master("local[*]") \
    .getOrCreate()

sc = spark.sparkContext

print("=" * 80)
print("Q1: RDD Basics")
print("=" * 80)

# ============================================================================
# METHOD 1: Create RDD from Python collection
# ============================================================================
print("\n--- METHOD 1: Create RDD from collection ---")

data = [1, 2, 3, 4, 5]
rdd = sc.parallelize(data)

print(f"RDD created: {rdd}")
print(f"Number of partitions: {rdd.getNumPartitions()}")

# ============================================================================
# METHOD 2: Transformations (lazy — not executed yet)
# ============================================================================
print("\n--- METHOD 2: Transformations (LAZY) ---")

# map() — transform each element
squared_rdd = rdd.map(lambda x: x ** 2)
print(f"After map(x^2): {squared_rdd}")

# filter() — keep only matching elements
even_squared = squared_rdd.filter(lambda x: x % 2 == 0)
print(f"After filter(even): {even_squared}")

# flatMap() — map then flatten
text_rdd = sc.parallelize(["Hello World", "PySpark Interview"])
words_rdd = text_rdd.flatMap(lambda x: x.split())
print(f"After flatMap(split): {words_rdd}")

print("\n✓ Note: None of the above executed yet — transformations are LAZY")

# ============================================================================
# METHOD 3: Actions (eager — trigger execution)
# ============================================================================
print("\n--- METHOD 3: Actions (TRIGGER EXECUTION) ---")

# Action: collect() — get all data to driver
result = squared_rdd.collect()
print(f"squared_rdd.collect(): {result}")
# Output: [1, 4, 9, 16, 25]

# Action: count() — count rows
count = even_squared.count()
print(f"even_squared.count(): {count}")
# Output: 2 (4 and 16)

# Action: first() — get first element
first_elem = rdd.first()
print(f"rdd.first(): {first_elem}")
# Output: 1

# Action: take(n) — get first n elements
first_three = rdd.take(3)
print(f"rdd.take(3): {first_three}")
# Output: [1, 2, 3]

# ============================================================================
# METHOD 4: Other transformations
# ============================================================================
print("\n--- METHOD 4: Other Transformations ---")

# union() — combine two RDDs
rdd1 = sc.parallelize([1, 2, 3])
rdd2 = sc.parallelize([4, 5, 6])
combined = rdd1.union(rdd2)
print(f"union result: {combined.collect()}")
# Output: [1, 2, 3, 4, 5, 6]

# distinct() — remove duplicates
dup_rdd = sc.parallelize([1, 2, 2, 3, 3, 3])
unique = dup_rdd.distinct()
print(f"distinct result: {unique.collect()}")
# Output: [1, 2, 3]

# reduceByKey() — aggregate by key (for pair RDDs)
pair_rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1), ("b", 1)])
count_by_key = pair_rdd.reduceByKey(lambda x, y: x + y)
print(f"reduceByKey result: {count_by_key.collect()}")
# Output: [('a', 2), ('b', 2)]

# groupByKey() — group by key
grouped = pair_rdd.groupByKey()
print(f"groupByKey result: {grouped.mapValues(list).collect()}")
# Output: [('a', [1, 1]), ('b', [1, 1])]

# ============================================================================
# METHOD 5: Compare RDD vs DataFrame
# ============================================================================
print("\n--- METHOD 5: RDD vs DataFrame ---")

# RDD approach (untyped)
rdd_people = sc.parallelize([
    (1, "Alice", 30),
    (2, "Bob", 25),
    (3, "Carol", 35)
])
print(f"RDD (untyped): {rdd_people.collect()}")

# DataFrame approach (typed with schema)
df_people = spark.createDataFrame([
    (1, "Alice", 30),
    (2, "Bob", 25),
    (3, "Carol", 35)
], ["id", "name", "age"])
print("DataFrame (typed):")
df_people.show()

# Performance: DataFrame will be faster due to Catalyst optimizer
# Optimization: Use DataFrame unless you need fine-grained control

# ============================================================================
# METHOD 6: RDD with custom partitions
# ============================================================================
print("\n--- METHOD 6: RDD with custom partitions ---")

# Create with specific number of partitions
rdd_partitioned = sc.parallelize(range(1, 11), numPartitions=4)
print(f"Partitions: {rdd_partitioned.getNumPartitions()}")

# Check data in each partition
def partition_data(partition):
    """Return partition index and count"""
    return [len(list(partition))]

partition_sizes = rdd_partitioned.mapPartitions(partition_data).collect()
print(f"Rows per partition: {partition_sizes}")

# ============================================================================
# METHOD 7: Persistence (caching)
# ============================================================================
print("\n--- METHOD 7: Persistence / Caching ---")

rdd_to_cache = sc.parallelize(range(1, 101))

# Cache for reuse
rdd_to_cache.cache()
print(f"Count 1st call (cached): {rdd_to_cache.count()}")
print(f"Count 2nd call (from cache): {rdd_to_cache.count()}")

# Check cache status
print(f"Cache status: {rdd_to_cache.getStorageLevel()}")

# Unpersist when done
rdd_to_cache.unpersist()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY — RDD Basics")
print("=" * 80)
print("""
✓ RDD Creation:
  - sc.parallelize() — from Python collection
  - sc.textFile() — from HDFS/S3
  - sc.wholeTextFiles() — read text files
  - sc.sequenceFile() — read sequence files
  
✓ Transformations (Lazy):
  - map() / flatMap() — transform elements
  - filter() — keep matching elements
  - distinct() — unique values
  - union() — combine RDDs
  - reduceByKey() / groupByKey() — aggregate
  
✓ Actions (Eager):
  - collect() — get all data to driver
  - count() — count elements
  - first() / take(n) — get sample
  - saveAsTextFile() / saveAsSequenceFile() — write
  
✓ Best Practice:
  - Prefer DataFrame/SQL API for better optimization
  - Use RDD for unstructured data or fine-grained control
  - Cache RDDs used multiple times
  - Avoid collect() on large RDDs (OOM risk)
""")

print("=" * 80)
