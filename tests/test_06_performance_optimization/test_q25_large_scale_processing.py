"""
Test for Q25 (Scenario): Large scale processing (1B+ rows)
"""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def test_q25_aggregation_pipeline(spark):
    """Test basic aggregation pipeline for large data."""
    data = spark.createDataFrame(
        [(i % 10, "Dept" + str(i % 5), i * 100) for i in range(1000)],
        ["id", "department", "amount"]
    )
    
    # Filter -> Group -> Aggregate
    result = data.filter(F.col("amount") > 0) \
        .groupBy("department") \
        .agg(F.sum("amount").alias("total_amount"))
    
    assert result.count() == 5  # 5 departments
    assert result.filter(F.col("total_amount") > 0).count() == 5


def test_q25_join_optimization(spark):
    """Test optimized join for large tables."""
    large_df = spark.createDataFrame(
        [(i, f"data{i % 10}") for i in range(1000)],
        ["id", "value"]
    )
    
    small_df = spark.createDataFrame(
        [(i, f"ref{i}") for i in range(10)],
        ["id", "reference"]
    )
    
    # Use broadcast for small table
    result = large_df.join(
        F.broadcast(small_df),
        on="id",
        how="left"
    )
    
    assert result.count() == 1000


def test_q25_partition_strategy(spark):
    """Test partitioning strategy for large data."""
    data = spark.createDataFrame(
        [(i, "2024-" + str(i % 12 + 1).zfill(2), i * 100)
         for i in range(1000)],
        ["id", "month", "amount"]
    )
    
    # Repartition for processing
    repartitioned = data.repartition(16, "month")
    
    assert repartitioned.rdd.getNumPartitions() == 16


def test_q25_cache_for_reuse(spark):
    """Test caching for multiple operations."""
    data = spark.createDataFrame(
        [(i, i % 10, i * 100) for i in range(1000)],
        ["id", "category", "value"]
    )
    
    # Cache data for reuse
    cached = data.cache()
    
    # Multiple operations on cached data
    count1 = cached.count()
    count2 = cached.filter(F.col("category") == 5).count()
    
    assert count1 == 1000
    assert count2 > 0
    
    # Cleanup
    cached.unpersist()


def test_q25_avoid_collect(spark):
    """Test avoiding unnecessary collect() for large data."""
    data = spark.createDataFrame(
        [(i, i * 100) for i in range(10000)],
        ["id", "value"]
    )
    
    # Use count() instead of collect().len()
    result_count = data.filter(F.col("value") > 50000).count()
    
    assert result_count > 0
    # Never do: result = data.collect() # <- Bad for large data!
