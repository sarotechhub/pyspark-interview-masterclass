"""
Test for Q23 (Scenario): Partitioned writes
"""

import pytest
from pyspark.sql import functions as F
import tempfile
import shutil
import os


def test_q23_partition_by_column(spark):
    """Test writing with partitionBy."""
    data = spark.createDataFrame(
        [(1, "2024-01", "A"), (2, "2024-01", "B"),
         (3, "2024-02", "C"), (4, "2024-02", "D")],
        ["id", "month", "value"]
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write partitioned by month
        data.write.mode("overwrite") \
            .partitionBy("month") \
            .parquet(tmpdir)
        
        # Read back to verify
        result = spark.read.parquet(tmpdir)
        
        assert result.count() == 4
        assert "month" in result.columns


def test_q23_coalesce_before_write(spark):
    """Test coalescing partitions before writing."""
    data = spark.createDataFrame(
        [(i, f"data{i}") for i in range(100)],
        ["id", "value"]
    )
    
    # Check default partitions
    original_partitions = data.rdd.getNumPartitions()
    
    # Coalesce to fewer partitions
    coalesced = data.coalesce(1)
    
    assert coalesced.rdd.getNumPartitions() == 1


def test_q23_incremental_write(spark):
    """Test incremental writes with new date."""
    import datetime
    
    today = datetime.date.today()
    
    data = spark.createDataFrame(
        [(1, "A", str(today)), (2, "B", str(today))],
        ["id", "value", "load_date"]
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate incremental write
        data.write.mode("overwrite") \
            .partitionBy("load_date") \
            .parquet(tmpdir)
        
        result = spark.read.parquet(tmpdir)
        
        assert result.filter(F.col("load_date") == str(today)).count() == 2


def test_q23_multiple_partition_columns(spark):
    """Test writing with multiple partition columns."""
    data = spark.createDataFrame(
        [(1, "2024-01", "USA", "A"),
         (2, "2024-01", "UK", "B"),
         (3, "2024-02", "USA", "C")],
        ["id", "month", "country", "value"]
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Partition by multiple columns
        data.write.mode("overwrite") \
            .partitionBy("month", "country") \
            .parquet(tmpdir)
        
        result = spark.read.parquet(tmpdir)
        
        assert result.count() == 3
