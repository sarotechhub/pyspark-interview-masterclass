"""
Test for Q19 (Scenario): SCD Type 2 (detect changes in data)
"""

import pytest
from pyspark.sql import functions as F


def test_q19_scd_type2_change_detection(spark):
    """Test detecting changes for SCD Type 2."""
    # Current snapshot
    current = spark.createDataFrame(
        [(1, "Alice", "NYC", "2024-01-01"),
         (2, "Bob", "LA", "2024-01-01")],
        ["id", "name", "city", "start_date"]
    )
    
    # New snapshot with changes
    new_data = spark.createDataFrame(
        [(1, "Alice", "Boston", "2024-02-01"),  # City changed
         (2, "Bob", "LA", "2024-01-01")],       # No change
        ["id", "name", "city", "start_date"]
    )
    
    # Join to detect changes
    comparison = current.alias("c").join(
        new_data.alias("n"),
        current.id == new_data.id,
        how="left_outer"
    ).withColumn(
        "changed",
        current.city != new_data.city
    )
    
    changed_count = comparison.filter(F.col("changed") == True).count()
    
    assert changed_count == 1
    assert comparison.count() == 2


def test_q19_add_version_number(spark):
    """Test adding version numbers for SCD Type 2."""
    data = spark.createDataFrame(
        [(1, "Alice", "NYC"), 
         (1, "Alice", "Boston"),  # Version 2
         (2, "Bob", "LA")],
        ["id", "name", "city"]
    )
    
    from pyspark.sql.window import Window
    
    window = Window.partitionBy("id").orderBy(F.monotonically_increasing_id())
    
    versioned = data.withColumn(
        "version",
        F.row_number().over(window)
    )
    
    alice_versions = versioned.filter(F.col("id") == 1).collect()
    
    assert alice_versions[0][3] == 1
    assert alice_versions[1][3] == 2
