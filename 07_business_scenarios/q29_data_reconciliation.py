"""
Q29: Reconcile two data sources — find mismatches row by row.

Scenario: Compare source and target tables after an ETL migration.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("Q29_DataReconciliation").master("local[*]").getOrCreate()

# ============================================
# SAMPLE DATA
# ============================================
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

print("=" * 60)
print("SOURCE TABLE:")
print("=" * 60)
source.show()

print("=" * 60)
print("TARGET TABLE:")
print("=" * 60)
target.show()

# ============================================
# SOLUTION
# ============================================

# 1. Missing from target
missing_in_target = source.join(target, on="id", how="left_anti")
print("\n" + "=" * 60)
print("MISSING IN TARGET:")
print("=" * 60)
missing_in_target.show()

# 2. Extra in target
extra_in_target = target.join(source, on="id", how="left_anti")
print("\n" + "=" * 60)
print("EXTRA IN TARGET:")
print("=" * 60)
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
print("\n" + "=" * 60)
print("VALUE MISMATCHES:")
print("=" * 60)
mismatch.show()

# Summary
print("\n" + "=" * 60)
print("RECONCILIATION SUMMARY:")
print("=" * 60)
print(f"Total source rows: {source.count()}")
print(f"Total target rows: {target.count()}")
print(f"Missing in target: {missing_in_target.count()}")
print(f"Extra in target: {extra_in_target.count()}")
print(f"Value mismatches: {mismatch.count()}")

spark.stop()
