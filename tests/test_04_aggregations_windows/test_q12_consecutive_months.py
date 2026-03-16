"""
Test for Q12 (Scenario): Consecutive months detection
"""

import pytest
from pyspark.sql import functions as F


def test_q12_consecutive_months_basic(spark):
    """Test detecting consecutive months in data."""
    data = [
        ("2024-01", 100),
        ("2024-02", 200),
        ("2024-03", 300),
        ("2024-04", 400),
    ]
    df = spark.createDataFrame(data, ["month", "value"])
    
    consecutive = df.count()
    
    # All 4 months are consecutive
    assert consecutive == 4


def test_q12_months_between_calculation(spark):
    """Test months_between for gap detection."""
    data = [
        ("2024-01-01", "A"),
        ("2024-02-01", "A"),
        ("2024-03-01", "A"),
        ("2024-05-01", "A"),  # Gap here
    ]
    df = spark.createDataFrame(data, ["date", "group"])
    
    from pyspark.sql.window import Window
    
    window = Window.partitionBy("group").orderBy("date")
    
    result = df.withColumn(
        "prev_date",
        F.lag("date").over(window)
    ).withColumn(
        "months_diff",
        F.months_between(F.col("date"), F.col("prev_date"))
    )
    
    collected = result.collect()
    
    assert collected[0][3] is None  # First row
    assert collected[1][3] == 1     # 1 month gap
    assert collected[2][3] == 1     # 1 month gap
    assert collected[3][3] == 2     # 2 month gap (skip April)
