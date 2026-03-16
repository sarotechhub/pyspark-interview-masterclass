"""
Test for Q5: Null Value Handling Strategies
"""

import pytest
from pyspark.sql import functions as F


def test_q05_drop_rows_with_nulls(spark):
    """Test dropna() to remove rows with null values."""
    data = [
        ("Alice", 25, 50000),
        ("Bob", None, 60000),
        ("Carol", 35, None),
        ("David", 40, 70000)
    ]
    df = spark.createDataFrame(data, ["name", "age", "salary"])
    
    # Drop rows with any null
    dropped = df.dropna()
    
    assert dropped.count() == 2  # Only Alice and David
    assert dropped.filter(F.col("name") == "Bob").count() == 0


def test_q05_drop_rows_any_vs_all(spark):
    """Test dropna(how='any') vs dropna(how='all')."""
    data = [
        ("Alice", 25, 50000),
        ("Bob", None, 60000),
        ("Carol", None, None),
        ("David", 40, 70000)
    ]
    df = spark.createDataFrame(data, ["name", "age", "salary"])
    
    # how='any': drop if ANY column is null
    drop_any = df.dropna(how="any")
    assert drop_any.count() == 2
    
    # how='all': drop only if ALL columns are null
    drop_all = df.dropna(how="all")
    assert drop_all.count() == 4  # None dropped (all columns have at least one value)


def test_q05_fill_nulls_with_value(spark):
    """Test fillna() to replace nulls with values."""
    data = [
        ("Alice", 25, None),
        ("Bob", None, 60000),
        ("Carol", 35, 70000)
    ]
    df = spark.createDataFrame(data, ["name", "age", "salary"])
    
    # Fill age nulls with 0, salary nulls with 50000
    filled = df.fillna({"age": 0, "salary": 50000})
    
    collected = filled.collect()
    assert collected[1][1] == 0  # Bob's age
    assert collected[0][2] == 50000  # Alice's salary


def test_q05_fill_nulls_with_forward_fill(spark):
    """Test forward fill strategy for nulls."""
    from pyspark.sql.window import Window
    
    data = [
        ("Alice", 2024, 1000),
        (None, 2024, 2000),
        ("Bob", None, 3000),
        ("Carol", 2026, 4000)
    ]
    df = spark.createDataFrame(data, ["name", "year", "amount"])
    
    # Forward fill: use last non-null value
    window = Window.orderBy("amount")
    filled = df.withColumn(
        "name",
        F.last(F.col("name"), ignorenulls=True).over(window)
    )
    
    assert filled.count() == 4


def test_q05_coalesce_multiple_columns(spark):
    """Test coalesce() to get first non-null value."""
    data = [
        (None, "Bob", "Charlie"),
        ("Alice", None, "Carol"),
        (None, None, "David")
    ]
    df = spark.createDataFrame(data, ["col1", "col2", "col3"])
    
    # Get first non-null value from columns
    result = df.withColumn(
        "first_non_null",
        F.coalesce(F.col("col1"), F.col("col2"), F.col("col3"))
    )
    
    collected = result.collect()
    assert collected[0][3] == "Bob"      # col1 is null, col2 has "Bob"
    assert collected[1][3] == "Alice"    # col1 has "Alice"
    assert collected[2][3] == "David"    # col1 and col2 null, col3 has "David"


def test_q05_filter_non_nulls(spark):
    """Test filtering out null values."""
    data = [
        ("Alice", 25),
        ("Bob", None),
        ("Carol", 35),
        ("David", None)
    ]
    df = spark.createDataFrame(data, ["name", "age"])
    
    # Keep only non-null ages
    non_null = df.filter(F.col("age").isNotNull())
    
    assert non_null.count() == 2
    assert non_null.filter(F.col("name") == "Bob").count() == 0


def test_q05_null_checks(spark):
    """Test various null checking methods."""
    data = [
        ("Alice", 25),
        ("Bob", None),
        ("Carol", 35)
    ]
    df = spark.createDataFrame(data, ["name", "age"])
    
    # isNull()
    nulls = df.filter(F.col("age").isNull())
    assert nulls.count() == 1
    
    # isNotNull()
    non_nulls = df.filter(F.col("age").isNotNull())
    assert non_nulls.count() == 2
