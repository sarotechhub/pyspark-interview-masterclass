"""
Test for Q24 (Scenario): Predicate pushdown
"""

import pytest
from pyspark.sql import functions as F


def test_q24_filter_early(spark):
    """Test applying filters early in the pipeline."""
    data = spark.createDataFrame(
        [(i, f"data{i}", "active" if i % 2 == 0 else "inactive")
         for i in range(1000)],
        ["id", "value", "status"]
    )
    
    # Filter early before other operations
    filtered = data.filter(F.col("status") == "active")
    result = filtered.select("id", "value")
    
    assert result.count() == 500


def test_q24_predicate_on_partition_column(spark):
    """Test filtering on partition columns."""
    data = spark.createDataFrame(
        [(1, "2024-01", "A"), (2, "2024-01", "B"),
         (3, "2024-02", "C"), (4, "2024-02", "D"),
         (5, "2024-03", "E")],
        ["id", "month", "value"]
    )
    
    # Filter on month (would be partition column)
    result = data.filter(F.col("month") == "2024-01")
    
    assert result.count() == 2


def test_q24_combine_filters(spark):
    """Test combining multiple predicates."""
    data = spark.createDataFrame(
        [(i, i % 10) for i in range(100)],
        ["id", "category"]
    )
    
    # Combine predicates for better pushdown
    result = data.filter(
        (F.col("id") > 10) & (F.col("category") == 5)
    )
    
    assert result.count() > 0


def test_q24_pushdown_order(spark):
    """Test that column selection follows filtering."""
    data = spark.createDataFrame(
        [(i, f"name{i}", f"email{i}", "active")
         for i in range(100)],
        ["id", "name", "email", "status"]
    )
    
    # Filter first, then select (more efficient)
    result = data.filter(F.col("status") == "active") \
        .select("id", "name")
    
    assert result.count() == 100
    assert len(result.columns) == 2
