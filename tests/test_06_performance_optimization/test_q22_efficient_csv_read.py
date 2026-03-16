"""
Test for Q22 (Scenario): Efficient CSV read for large files
"""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType


def test_q22_define_schema(spark):
    """Test defining schema for efficient CSV read."""
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("age", IntegerType(), True),
    ])
    
    # Create test data
    data = [(1, "Alice", 25), (2, "Bob", 30)]
    df = spark.createDataFrame(data, schema=schema)
    
    assert df.count() == 2
    assert len(df.columns) == 3


def test_q22_csv_options(spark):
    """Test CSV reading with options."""
    from pyspark.sql.types import StructType, StructField, StringType
    
    schema = StructType([
        StructField("id", StringType()),
        StructField("value", StringType()),
    ])
    
    # Simulate CSV options
    data = [(1, "A"), (2, "B")]
    df = spark.createDataFrame(data, schema=schema)
    
    # Options like header=True, inferSchema=False should be used
    assert not df.isEmpty()


def test_q22_avoid_infer_schema(spark):
    """Test impact of avoiding inferSchema."""
    data = [(1, "A"), (2, "B"), (3, "C")] * 10
    
    # Without schema (would infer)
    df_no_schema = spark.createDataFrame(data, ["id", "value"])
    
    # With schema (preferred)
    from pyspark.sql.types import StructType, StructField, IntegerType, StringType
    
    schema = StructType([
        StructField("id", IntegerType()),
        StructField("value", StringType()),
    ])
    
    df_with_schema = spark.createDataFrame(data, schema=schema)
    
    # Both should have same count but with_schema is more efficient
    assert df_no_schema.count() == df_with_schema.count()


def test_q22_select_needed_columns(spark):
    """Test selecting only needed columns."""
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType
    
    schema = StructType([
        StructField("id", IntegerType()),
        StructField("name", StringType()),
        StructField("email", StringType()),
        StructField("phone", StringType()),
    ])
    
    data = [
        (1, "Alice", "alice@example.com", "555-1234"),
        (2, "Bob", "bob@example.com", "555-5678")
    ]
    
    df = spark.createDataFrame(data, schema=schema)
    
    # Select only needed columns
    selected = df.select("id", "name")
    
    assert len(selected.columns) == 2
    assert selected.count() == 2
