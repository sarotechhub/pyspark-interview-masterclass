"""
Test for Q5 (Scenario): Forward fill nulls
"""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def test_q05_forward_fill_basic(spark):
    """Test forward fill with last non-null value."""
    data = [
        ("2024-01-01", "A"),
        ("2024-01-02", None),
        ("2024-01-03", None),
        ("2024-01-04", "B"),
        ("2024-01-05", None),
    ]
    df = spark.createDataFrame(data, ["date", "value"])
    
    # Forward fill: use last non-null value
    window = Window.orderBy("date") \
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    
    filled = df.withColumn(
        "filled_value",
        F.last(F.col("value"), ignorenulls=True).over(window)
    )
    
    collected = filled.collect()
    
    assert collected[0][2] == "A"      # First: A
    assert collected[1][2] == "A"      # Second: A (forward fill)
    assert collected[2][2] == "A"      # Third: A (forward fill)
    assert collected[3][2] == "B"      # Fourth: B
    assert collected[4][2] == "B"      # Fifth: B (forward fill)


def test_q05_forward_fill_per_group(spark):
    """Test forward fill within groups."""
    data = [
        ("Group1", 1, "X"),
        ("Group1", 2, None),
        ("Group1", 3, None),
        ("Group2", 1, "Y"),
        ("Group2", 2, None),
    ]
    df = spark.createDataFrame(data, ["group", "order", "value"])
    
    window = Window.partitionBy("group").orderBy("order") \
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    
    filled = df.withColumn(
        "filled",
        F.last(F.col("value"), ignorenulls=True).over(window)
    )
    
    group1_filled = filled.filter(F.col("group") == "Group1").collect()
    
    assert group1_filled[0][3] == "X"
    assert group1_filled[1][3] == "X"
    assert group1_filled[2][3] == "X"


def test_q05_forward_fill_verification(spark):
    """Test counting nulls before and after forward fill."""
    data = [
        ("A", None),
        ("B", None),
        ("C", "Value"),
        ("D", None),
    ]
    df = spark.createDataFrame(data, ["id", "status"])
    
    # Count nulls before
    nulls_before = df.filter(F.col("status").isNull()).count()
    assert nulls_before == 3
    
    # Forward fill
    window = Window.orderBy("id") \
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    
    filled = df.withColumn(
        "filled_status",
        F.last(F.col("status"), ignorenulls=True).over(window)
    )
    
    # Count nulls after in filled column
    nulls_after = filled.filter(F.col("filled_status").isNull()).count()
    assert nulls_after < nulls_before
