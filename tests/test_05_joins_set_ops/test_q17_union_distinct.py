"""
Test for Q17 (Scenario): Union and distinct operations
"""

import pytest
from pyspark.sql import functions as F


def test_q17_union_basic(spark):
    """Test union of two DataFrames."""
    df1 = spark.createDataFrame(
        [(1, "Alice"), (2, "Bob")],
        ["id", "name"]
    )
    df2 = spark.createDataFrame(
        [(3, "Carol"), (4, "David")],
        ["id", "name"]
    )
    
    # Union combines all rows
    result = df1.union(df2)
    
    assert result.count() == 4


def test_q17_union_distinct(spark):
    """Test union followed by distinct to remove duplicates."""
    df1 = spark.createDataFrame(
        [(1, "Alice"), (2, "Bob"), (1, "Alice")],  # Duplicate
        ["id", "name"]
    )
    df2 = spark.createDataFrame(
        [(1, "Alice"), (3, "Carol")],
        ["id", "name"]
    )
    
    # Union then distinct
    result = df1.union(df2).distinct()
    
    assert result.count() == 3  # Alice, Bob, Carol
    
    # Verify no duplicates
    alice_count = result.filter(F.col("name") == "Alice").count()
    assert alice_count == 1


def test_q17_union_by_name(spark):
    """Test unionByName for column-based union."""
    df1 = spark.createDataFrame(
        [(1, "Alice", 25)],
        ["id", "name", "age"]
    )
    df2 = spark.createDataFrame(
        [(2, "Bob", 30)],
        ["id", "name", "age"]
    )
    
    # unionByName matches columns by name
    result = df1.unionByName(df2)
    
    assert result.count() == 2
    assert result.columns == ["id", "name", "age"]


def test_q17_drop_duplicates(spark):
    """Test dropDuplicates for removing duplicate rows."""
    data = spark.createDataFrame(
        [(1, "A"), (2, "B"), (1, "A"), (3, "C"), (2, "B")],
        ["id", "value"]
    )
    
    # Remove duplicates
    unique = data.dropDuplicates()
    
    assert unique.count() == 3


def test_q17_drop_duplicates_subset(spark):
    """Test dropDuplicates on specific columns."""
    data = spark.createDataFrame(
        [(1, "A", "X"), (1, "A", "Y"), (2, "B", "Z")],
        ["id", "val1", "val2"]
    )
    
    # Remove duplicates based on id and val1 only
    unique = data.dropDuplicates(["id", "val1"])
    
    assert unique.count() == 2
