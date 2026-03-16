"""
Test for Q29 (Scenario): Data reconciliation
"""

import pytest
from pyspark.sql import functions as F


def test_q29_find_missing_records(spark):
    """Test finding records in one dataset but not in another."""
    system_a = spark.createDataFrame(
        [(1, "Alice"), (2, "Bob"), (3, "Carol")],
        ["id", "name"]
    )
    
    system_b = spark.createDataFrame(
        [(1, "Alice"), (2, "Bob"), (4, "David")],
        ["id", "name"]
    )
    
    # Records in A but not in B
    missing_in_b = system_a.join(system_b, "id", how="left_anti")
    
    assert missing_in_b.count() == 1
    assert missing_in_b.collect()[0][0] == 3


def test_q29_find_mismatches(spark):
    """Test finding records with different values."""
    source = spark.createDataFrame(
        [(1, "Alice", 50000),
         (2, "Bob", 60000),
         (3, "Carol", 55000)],
        ["id", "name", "salary"]
    )
    
    target = spark.createDataFrame(
        [(1, "Alice", 50000),
         (2, "Bob", 65000),  # Mismatch
         (3, "Carol", 55000)],
        ["id", "name", "salary"]
    )
    
    # Find mismatches
    comparison = source.alias("src").join(
        target.alias("tgt"),
        source.id == target.id,
        how="inner"
    ).filter(
        source.salary != target.salary
    )
    
    mismatches = comparison.count()
    
    assert mismatches == 1


def test_q29_reconciliation_summary(spark):
    """Test generating reconciliation summary."""
    source = spark.createDataFrame(
        [(1, 100), (2, 200), (3, 300)],
        ["id", "amount"]
    )
    
    target = spark.createDataFrame(
        [(1, 100), (2, 200), (4, 400)],
        ["id", "amount"]
    )
    
    # Count records
    source_count = source.count()
    target_count = target.count()
    
    # Find matching records
    matched = source.join(target, "id", how="inner").count()
    
    # Reconciliation summary
    assert source_count == 3
    assert target_count == 3
    assert matched == 2


def test_q29_duplicate_detection(spark):
    """Test detecting duplicate records in data."""
    data = spark.createDataFrame(
        [(1, "Alice", "alice@example.com"),
         (2, "Bob", "bob@example.com"),
         (1, "Alice", "alice@example.com"),  # Duplicate
         (3, "Carol", "carol@example.com"),
         (2, "Bob", "bob@example.com")],  # Duplicate
        ["id", "name", "email"]
    )
    
    # Find duplicates
    duplicates = data.groupBy("id", "name", "email").count() \
        .filter(F.col("count") > 1)
    
    assert duplicates.count() == 2
