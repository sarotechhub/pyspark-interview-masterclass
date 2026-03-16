"""
Test for Q3: SparkSession — Configuration & Session Management
"""

import pytest
from pyspark.sql import SparkSession, functions as F


def test_q03_sparksession_creation(spark):
    """Test SparkSession is created and configured."""
    assert spark is not None
    assert spark.appName == "pytest-pyspark"


def test_q03_sparksession_create_dataframe(spark):
    """Test creating a DataFrame through SparkSession."""
    data = [("Alice", 25), ("Bob", 30), ("Carol", 35)]
    df = spark.createDataFrame(data, ["name", "age"])
    
    assert df.count() == 3
    assert df.columns == ["name", "age"]
    
    # Verify data
    collected = df.collect()
    assert collected[0][0] == "Alice"
    assert collected[1][1] == 30


def test_q03_sparksession_sql_execution(spark):
    """Test SQL execution through SparkSession."""
    data = [(1, "Alice"), (2, "Bob"), (3, "Carol")]
    df = spark.createDataFrame(data, ["id", "name"])
    
    # Register temporary view
    df.createOrReplaceTempView("people")
    
    # Execute SQL
    result = spark.sql("SELECT * FROM people WHERE id > 1")
    
    assert result.count() == 2
    assert result.select("name").collect()[0][0] == "Bob"


def test_q03_sparksession_config(spark):
    """Test SparkSession configuration."""
    # Verify config settings
    shuffle_partitions = spark.conf.get("spark.sql.shuffle.partitions")
    assert int(shuffle_partitions) == 8


def test_q03_extract_sparkcontext(spark):
    """Test accessing SparkContext from SparkSession."""
    sc = spark.sparkContext
    
    assert sc is not None
    
    # Create RDD via SparkContext
    rdd = sc.parallelize([1, 2, 3, 4, 5])
    assert rdd.count() == 5


def test_q03_dataframe_schema(spark):
    """Test DataFrame schema definition."""
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType
    
    schema = StructType([
        StructField("name", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("salary", IntegerType(), True)
    ])
    
    data = [("Alice", 25, 50000), ("Bob", 30, 60000)]
    df = spark.createDataFrame(data, schema=schema)
    
    assert df.count() == 2
    assert len(df.columns) == 3
    assert df.schema.fields[0].name == "name"


def test_q03_dataframe_basic_operations(spark):
    """Test basic DataFrame operations."""
    data = [("Alice", 25, 50000), ("Bob", 30, 60000), ("Carol", 35, 70000)]
    df = spark.createDataFrame(data, ["name", "age", "salary"])
    
    # Select operation
    selected = df.select("name", "age")
    assert selected.columns == ["name", "age"]
    
    # Filter operation
    filtered = df.filter(F.col("age") > 25)
    assert filtered.count() == 2
    
    # WithColumn operation
    with_bonus = df.withColumn("bonus", F.col("salary") * 0.1)
    assert "bonus" in with_bonus.columns
