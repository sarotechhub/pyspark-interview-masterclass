"""
Q19: Slowly Changing Dimension Type 2 — detect new/changed records.

Scenario: Daily snapshot arrives. Detect what changed since yesterday for SCD Type 2.
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - join(how='left_outer'): O(n+m) — hash join
#   - filter() for changes: O(n) — row-wise comparison
#   - Total: O(n+m) — linear in table sizes
#
# Shuffle Operations:
#   - join(): FULL SHUFFLE (redistribute both tables)
#   - left_outer join larger table (current)
#
# Performance Tips:
#   - left_outer join keeps all from left (current) table
#   - Filter on null right side identifies new records
#   - isNull() check on join column identifies missing
#   - Column comparison can be expensive (multiple columns)
#   - Consider: hash comparison if many columns
#   - Cache current state table (reused every day)
#   - Partition by effective date for time series queries
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Q19_SCD_Type2") \
    .master("local[*]") \
    .getOrCreate()

# ============================================
# SAMPLE DATA
# ============================================
# Historical data (from previous load)
current = spark.createDataFrame([
    (1, "Alice", "NY",  "2024-01-01"),
    (2, "Bob",   "CA",  "2024-01-01"),
    (3, "Carol", "TX",  "2024-02-01"),
], ["id", "name", "city", "effective_date"])

# New snapshot (today's data)
new_snapshot = spark.createDataFrame([
    (1, "Alice", "TX"),   # City changed NY → TX
    (2, "Bob",   "CA"),   # No change
    (3, "Carol", "FL"),   # City changed TX → FL
    (4, "Dave",  "NY"),   # New record
], ["id", "name", "city"])

print("=" * 60)
print("CURRENT STATE (Previous load):")
print("=" * 60)
current.show()

print("=" * 60)
print("NEW SNAPSHOT (Today's data - no historical columns):")
print("=" * 60)
new_snapshot.show()

# ============================================
# SOLUTION
# ============================================

# Step 1: Left outer join to find new or changed records
#         NEW: id not in current
#         CHANGED: id in current but some column differs
changed = new_snapshot.alias("new") \
    .join(current.alias("curr"), on="id", how="left") \
    .filter(
        F.col("curr.id").isNull() |                      # New record
        (F.col("curr.city") != F.col("new.city"))        # Changed record
    ) \
    .select(
        F.col("new.id"),
        F.col("new.name"),
        F.col("new.city"),
        F.current_date().alias("effective_date"),
        "curr.effective_date"
    )

print("\n" + "=" * 60)
print("DETECTED CHANGES (new or modified records):")
print("=" * 60)
changed.show()

# ============================================
# BUILD SCD TYPE 2 TABLE
# ============================================

# For SCD Type 2, we need to:
# 1. Mark old records as inactive (end_date = today - 1)
# 2. Insert new active records

# Step 1: Mark records as ended if they were changed
ended_records = current.alias("c") \
    .join(changed.alias("ch"), on=F.col("c.id") == F.col("ch.id"), how="left_semi") \
    .withColumn("end_date", F.current_date())

# Step 2: Insert new records (both entirely new AND changes)
new_records = changed.select(
    "id", "name", "city",
    F.current_date().alias("effective_date"),
    F.lit(None).alias("end_date").cast("date"),
    F.lit(True).alias("is_active")
)

# Step 3: Keep unchanged current records
unchanged = new_snapshot.alias("new") \
    .join(current.alias("curr"), on="id", how="inner") \
    .filter(F.col("curr.city") == F.col("new.city"))

unchanged_records = unchanged \
    .select(
        F.col("curr.id"),
        F.col("curr.name"),
        F.col("curr.city"),
        F.col("curr.effective_date"),
        F.lit(None).alias("end_date").cast("date"),
        F.lit(True).alias("is_active")
    )

# Step 4: Combine all
scd_updates = spark.createDataFrame([], 
    "id INT, name STRING, city STRING, effective_date DATE, end_date DATE, is_active BOOLEAN")

# Union: ended records + unchanged records + new records
if ended_records.count() > 0:
    scd_updates = ended_records.union(unchanged_records).union(new_records)
else:
    scd_updates = unchanged_records.union(new_records)

print("\n" + "=" * 60)
print("COMPLETE SCD TYPE 2 STATE:")
print("=" * 60)
scd_updates.select("id", "name", "city", "effective_date", "end_date", "is_active").show()

# ============================================
# ALTERNATIVE: Using MERGE (Delta Lake)
# ============================================
# Note: This would use Delta Lake's MERGE statement
# Example (pseudocode):
# MERGE INTO scd_table AS target
# USING new_snapshot AS source
# WHEN MATCHED AND target.city != source.city THEN
#   UPDATE SET end_date = current_date, is_active = false
# WHEN NOT MATCHED THEN
#   INSERT (id, name, city, effective_date, is_active)
#   VALUES (source.id, source.name, source.city, current_date, true)

# ============================================
# SUMMARY REPORT
# ============================================
print("\n" + "=" * 60)
print("CHANGE SUMMARY:")
print("=" * 60)
print(f"Total records in new snapshot: {new_snapshot.count()}")
print(f"New records (not in current): {changed.where(F.col('curr.id').isNull()).count()}")
print(f"Changed records: {changed.where(F.col('curr.id').isNotNull()).count()}")
print(f"Unchanged records: {unchanged.count()}")

# ============================================
# EXPECTED OUTPUT (Changes):
# ============================================
# +---+-----+-----+----------------+--+
# | id| name| city|effective_date  |.. |
# +---+-----+-----+----------------+--+
# |  1|Alice|TX   |2024-XX-XX      |.. |
# |  3|Carol|FL   |2024-XX-XX      |.. |
# |  4|Dave |NY   |2024-XX-XX      |.. |
# +---+-----+-----+----------------+--+

spark.stop()
