"""
Q2: Transformations vs Actions — Understand lazy vs eager execution

Scenario: Learn the difference between transformations and actions.
Transformations are lazy (not executed immediately), while actions trigger execution.

Key Concepts:
- Transformations: Create new RDD/DataFrame (map, filter, groupBy, select, etc.)
- Actions: Return results or write to storage (collect, show, count, write, etc.)
- Lazy Evaluation: Allows Spark to optimize the entire pipeline before execution
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - Transformations: O(1) — lazy, no execution cost
#   - Actions: Depends on logic (collect O(n), show O(limit), count O(n))
#   - Filter + Collect: O(n) — processes all rows
#   - GroupBy + Agg: O(n + k) — shuffle then aggregate
#
# Shuffle Operations:
#   - filter(): NO SHUFFLE (element-wise operation)
#   - select(): NO SHUFFLE (column extraction)
#   - groupBy(): FULL SHUFFLE (redistributes by key)
#   - join(): FULL SHUFFLE (repartition both sides)
#
# Performance Tips:
#   - Chain multiple transformations (lazy optimization)
#   - Filter early to reduce data before shuffle operations
#   - Use select() before join() to reduce column count
#   - Avoid multiple actions on same DataFrame (cache if needed)
#   - Use count() sparingly — triggers full computation
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Q02_Transformations_Actions") \
    .master("local[*]") \
    .getOrCreate()

sc = spark.sparkContext

print("=" * 80)
print("Q2: Transformations vs Actions")
print("=" * 80)

# ============================================================================
# Sample Data
# ============================================================================
data = [
    (1, "Alice", 50000),
    (2, "Bob", 75000),
    (3, "Carol", 60000),
    (4, "Dave", 95000),
    (5, "Eve", 55000),
]

df = spark.createDataFrame(data, ["id", "name", "salary"])

# ============================================================================
# METHOD 1: Understanding Transformations (LAZY)
# ============================================================================
print("\n--- METHOD 1: Transformations (LAZY - Not executed) ---")

# These lines create transformation plans but DON'T execute
print("Creating transformation 1: filter(salary > 60000)")
filtered = df.filter(F.col("salary") > 60000)
print(f"  → {filtered} (plan created, NOT executed)")

print("\nCreating transformation 2: select(name, salary)")
selected = filtered.select("name", "salary")
print(f"  → {selected} (plan created, NOT executed)")

print("\nCreating transformation 3: withColumn(salary_doubled)")
with_col = selected.withColumn("salary_doubled", F.col("salary") * 2)
print(f"  → {with_col} (plan created, NOT executed)")

print("\n✓ Note: All above transformations are LAZY. No data processed yet!")

# ============================================================================
# METHOD 2: Triggering Actions (EAGER)
# ============================================================================
print("\n--- METHOD 2: Actions (EAGER - Trigger execution) ---")

print("\nAction 1: .show() — Display results")
print("  Result:")
with_col.show()

print("\nAction 2: .count() — Count rows")
count = with_col.count()
print(f"  Result: {count} rows")

print("\nAction 3: .collect() — Get all data to driver")
result = with_col.collect()
print(f"  Result: {result}")

print("\nAction 4: .first() — Get first row")
first = with_col.first()
print(f"  Result: {first}")

print("\nAction 5: .take(n) — Get first n rows")
taken = with_col.take(2)
print(f"  Result: {taken}")

# ============================================================================
# METHOD 3: DataFrame Transformations Examples
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 3: Common DataFrame Transformations ---")
print("=" * 80)

df.show()

# 1. select() — choose/transform columns
print("\n1. select():")
df.select("name", "salary").show()

# 2. withColumn() — add/replace column
print("\n2. withColumn():")
df.withColumn("salary_category", 
    F.when(F.col("salary") > 80000, "Senior")
     .when(F.col("salary") > 60000, "Middle")
     .otherwise("Junior")
).show()

# 3. filter() — keep rows matching condition
print("\n3. filter():")
df.filter(F.col("salary") > 60000).show()

# 4. groupBy() + agg() — aggregate
print("\n4. groupBy() + agg():")
df.groupBy().agg(F.avg("salary").alias("avg_salary")).show()

# 5. alias() — rename column
print("\n5. alias() + select():")
df.select(F.col("name").alias("employee_name")).show()

# 6. cast() — convert data type
print("\n6. cast() — convert to string:")
df.select(F.col("salary").cast(StringType()).alias("salary_str")).show()

# 7. orderBy() — sort
print("\n7. orderBy():")
df.orderBy(F.desc("salary")).show()

# 8. distinct() — unique rows
print("\n8. distinct():")
df.select("name").distinct().show()

# 9. limit() — take first n rows
print("\n9. limit():")
df.limit(2).show()

# 10. dropDuplicates() — remove duplicates
print("\n10. dropDuplicates():")
df.dropDuplicates(["id"]).show()

# ============================================================================
# METHOD 4: RDD Transformations Examples
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 4: Common RDD Transformations ---")
print("=" * 80)

rdd = sc.parallelize([1, 2, 3, 4, 5])

# 1. map() — transform each element
print("\n1. map(x * 2):")
mapped = rdd.map(lambda x: x * 2)
print(f"  Result: {mapped.collect()}")

# 2. filter() — keep matching elements
print("\n2. filter(x > 2):")
filtered = rdd.filter(lambda x: x > 2)
print(f"  Result: {filtered.collect()}")

# 3. flatMap() — map then flatten
print("\n3. flatMap():")
text_rdd = sc.parallelize(["Hello World", "PySpark"])
words = text_rdd.flatMap(lambda x: x.split())
print(f"  Result: {words.collect()}")

# 4. union() — combine RDDs
rdd1 = sc.parallelize([1, 2])
rdd2 = sc.parallelize([3, 4])
print("\n4. union():")
print(f"  Result: {rdd1.union(rdd2).collect()}")

# 5. distinct() — remove duplicates
print("\n5. distinct():")
dup_rdd = sc.parallelize([1, 1, 2, 2, 3])
print(f"  Result: {dup_rdd.distinct().collect()}")

# 6. reduceByKey()
print("\n6. reduceByKey():")
pairs = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
print(f"  Result: {pairs.reduceByKey(lambda x, y: x + y).collect()}")

# ============================================================================
# METHOD 5: Actions Examples
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 5: Common Actions ---")
print("=" * 80)

# 1. show() — display
print("\n1. df.show():")
df.show(2)

# 2. collect() — get all data
print("\n2. df.collect():")
result = df.select("name").limit(2).collect()
print(f"  {result}")

# 3. count() — count rows
print("\n3. df.count():")
print(f"  {df.count()}")

# 4. first() — first row
print("\n4. df.first():")
print(f"  {df.first()}")

# 5. take(n) — first n rows
print("\n5. df.take(2):")
print(f"  {df.take(2)}")

# 6. describe() — statistics
print("\n6. df.describe():")
df.describe("salary").show()

# 7. write — save to file
print("\n7. df.write.parquet() / df.write.csv():")
print("  (Would write to /tmp/output.parquet)")
# df.write.mode("overwrite").parquet("/tmp/output")

# 8. rdd.saveAsTextFile()
print("\n8. rdd.saveAsTextFile():")
print("  (Would save RDD to disk)")
# rdd.saveAsTextFile("/tmp/output")

# ============================================================================
# METHOD 6: Demonstrating Lazy Evaluation Benefits
# ============================================================================
print("\n" + "=" * 80)
print("--- METHOD 6: Lazy Evaluation Optimization ---")
print("=" * 80)

print("\nWithout lazy evaluation, would execute:")
print("  1. Load 1M rows")
print("  2. Filter → 100K rows")
print("  3. Select columns → 100K rows")

print("\nWith lazy evaluation, Spark optimizes and execute:")
print("  1. Load + Filter + Select (combined in one pass)")
print("  2. Only needed columns are read (column pruning)")
print("  3. Only matching rows are kept (predicate pushdown)")

# Build a long chain (still lazy)
result = df \
    .filter(F.col("salary") > 50000) \
    .select("id", "name", "salary") \
    .withColumn("bonus", F.col("salary") * 0.1) \
    .filter(F.col("bonus") > 5000)

print("\nFinal result (now executing the entire optimized plan):")
result.show()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY — Transformations vs Actions")
print("=" * 80)
print("""
✓ TRANSFORMATIONS (LAZY):
  - Create new DataFrame/RDD but don't execute
  - Can be chained together
  - DataFrame: select, withColumn, filter, groupBy, orderBy, distinct, etc.
  - RDD: map, filter, flatMap, union, distinct, reduceByKey, groupByKey, etc.

✓ ACTIONS (EAGER):
  - Trigger execution immediately
  - Return results or write to storage
  - DataFrame: show, collect, count, first, take, write, describe, etc.
  - RDD: collect, count, first, saveAsTextFile, saveAsSequenceFile, etc.

✓ KEY BENEFITS OF LAZY EVALUATION:
  1. Optimization: Spark optimizes entire pipeline before execution
  2. Performance: Unnecessary operations are eliminated
  3. Memory: Only needed data is computed
  4. Pushdown: Filters pushed to data source when possible

✓ BEST PRACTICES:
  - Chain transforms together before action
  - Avoid calling actions inside loops
  - Use cache() for DataFrames used multiple times
  - Use collect() only on small results (OOM risk on large data)
""")

print("=" * 80)
