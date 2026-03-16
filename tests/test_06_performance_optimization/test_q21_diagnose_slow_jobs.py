"""
Test for Q21 (Scenario): Diagnose slow jobs
"""

import pytest
from pyspark.sql import functions as F


def test_q21_explain_plan(spark):
    """Test using explain() to diagnose query execution."""
    data = spark.createDataFrame(
        [(1, "A"), (2, "B"), (3, "C")],
        ["id", "value"]
    )
    
    # Create a query
    result = data.filter(F.col("id") > 1).select("value")
    
    # Get execution plan (should not error)
    plan = result.explain(extended=True)
    
    # Just verify it doesn't error
    assert result.count() == 2


def test_q21_partition_diagnostics(spark):
    """Test checking partition count."""
    data = spark.createDataFrame(
        [(i, f"data{i}") for i in range(100)],
        ["id", "value"]
    )
    
    # Check default partitions
    partitions = data.rdd.getNumPartitions()
    assert partitions > 0
    
    # Repartition and check
    repartitioned = data.repartition(4)
    new_partitions = repartitioned.rdd.getNumPartitions()
    
    assert new_partitions == 4


def test_q21_data_skew_detection(spark):
    """Test detecting data skew."""
    skewed_data = spark.createDataFrame(
        [(1, "A")] * 1000 +  # Most data in partition 1
        [(2, "B")] * 10 +
        [(3, "C")] * 5,
        ["id", "value"]
    )
    
    # Count per id to identify skew
    skew_analysis = skewed_data.groupBy("id").count() \
        .withColumn(
            "pct_of_total",
            (F.col("count") / F.sum("count").over(F.Window.partitionBy())) * 100
        )
    
    collected = skew_analysis.collect()
    
    # First partition should have > 90% of data
    first_row_pct = collected[0][2]
    assert first_row_pct > 90
