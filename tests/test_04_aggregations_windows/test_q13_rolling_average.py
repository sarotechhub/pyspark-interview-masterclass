"""
Test for Q13 (Scenario): 7-day rolling average
"""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def test_q13_rolling_average_basic(spark):
    """Test calculating 7-day rolling average."""
    data = [
        (1, 100), (2, 110), (3, 105), (4, 115),
        (5, 120), (6, 125), (7, 130),
        (8, 135), (9, 140),
    ]
    df = spark.createDataFrame(data, ["day", "value"])
    
    # 7-day rolling average
    window = Window.orderBy("day") \
        .rowsBetween(-6, 0)  # Current row and 6 rows back
    
    result = df.withColumn(
        "rolling_avg_7day",
        F.round(F.avg("value").over(window), 2)
    )
    
    collected = result.collect()
    
    # Day 1-6 have partial windows
    assert collected[0][2] is not None
    assert collected[5][2] is not None
    
    # Calculate day 7 average manually: (100+110+105+115+120+125+130)/7
    expected_day7 = round((100+110+105+115+120+125+130)/7, 2)
    assert collected[6][2] == expected_day7


def test_q13_rolling_average_per_group(spark):
    """Test rolling average within groups."""
    data = [
        ("A", 1, 100), ("A", 2, 110), ("A", 3, 120),
        ("B", 1, 200), ("B", 2, 210), ("B", 3, 220),
    ]
    df = spark.createDataFrame(data, ["group", "day", "value"])
    
    window = Window.partitionBy("group").orderBy("day") \
        .rowsBetween(-1, 0)  # 2-day average
    
    result = df.withColumn(
        "rolling_avg",
        F.avg("value").over(window)
    )
    
    collected = result.collect()
    
    # Group A: values 100, 110, 120
    group_a = [row for row in collected if row[0] == "A"]
    
    assert group_a[0][3] == 100        # First row (only one value)
    assert group_a[1][3] == 105        # (100+110)/2
