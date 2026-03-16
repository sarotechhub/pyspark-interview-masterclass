"""
Test for Q20 (Scenario): Skewed join optimization
"""

import pytest
from pyspark.sql import functions as F


def test_q20_broadcast_join(spark):
    """Test broadcast join for small tables."""
    large_df = spark.createDataFrame(
        [(i, f"value{i}") for i in range(1000)],
        ["id", "data"]
    )
    
    small_df = spark.createDataFrame(
        [(i, f"small{i}") for i in range(1, 10)],
        ["id", "small_data"]
    )
    
    # Broadcast small table
    result = large_df.join(
        F.broadcast(small_df),
        on="id",
        how="inner"
    )
    
    assert result.count() == 9


def test_q20_join_without_broadcast(spark):
    """Test regular join for comparison."""
    df1 = spark.createDataFrame(
        [(1, "A"), (2, "B"), (3, "C")],
        ["id", "val1"]
    )
    
    df2 = spark.createDataFrame(
        [(1, "X"), (2, "Y")],
        ["id", "val2"]
    )
    
    # Regular join
    result = df1.join(df2, "id", how="inner")
    
    assert result.count() == 2
    assert result.columns == ["id", "val1", "val2"]


def test_q20_skew_detection(spark):
    """Test identifying skewed joins."""
    data = spark.createDataFrame(
        [(1, "A")] * 10000 +  # Heavy skew towards id=1
        [(2, "B")] * 10 +
        [(3, "C")] * 5,
        ["id", "value"]
    )
    
    # Count per id to detect skew
    count_by_id = data.groupBy("id").count()
    
    collected = count_by_id.collect()
    
    # Find the most frequent id
    max_count = max(row[1] for row in collected)
    assert max_count == 10000  # Significant skew
