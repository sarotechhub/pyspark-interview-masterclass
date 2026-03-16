"""
Test for Q1: RDD Basics — Create, transform, and count RDDs
"""

import pytest
from pyspark.sql import functions as F


def test_q01_rdd_creation(spark_context):
    """Test basic RDD creation and partitions."""
    data = [1, 2, 3, 4, 5]
    rdd = spark_context.parallelize(data)
    
    assert rdd.count() == 5
    assert rdd.getNumPartitions() > 0
    assert rdd.collect() == [1, 2, 3, 4, 5]


def test_q01_rdd_map_transformation(spark_context):
    """Test map transformation on RDD."""
    data = [1, 2, 3, 4, 5]
    rdd = spark_context.parallelize(data)
    
    # Test map: square each element
    squared_rdd = rdd.map(lambda x: x ** 2)
    result = squared_rdd.collect()
    
    assert result == [1, 4, 9, 16, 25]


def test_q01_rdd_filter_transformation(spark_context):
    """Test filter transformation on RDD."""
    data = [1, 2, 3, 4, 5]
    rdd = spark_context.parallelize(data)
    
    # Test filter: keep only even numbers
    even_rdd = rdd.filter(lambda x: x % 2 == 0)
    result = even_rdd.collect()
    
    assert result == [2, 4]


def test_q01_rdd_flatmap_transformation(spark_context):
    """Test flatMap transformation on RDD."""
    data = ["Hello World", "PySpark Interview"]
    rdd = spark_context.parallelize(data)
    
    # Test flatMap: split words and flatten
    words_rdd = rdd.flatMap(lambda x: x.split())
    result = words_rdd.collect()
    
    assert len(result) == 4
    assert "Hello" in result
    assert "PySpark" in result


def test_q01_rdd_count_action(spark_context):
    """Test count action on RDD."""
    data = [1, 2, 3, 4, 5]
    rdd = spark_context.parallelize(data)
    
    count = rdd.count()
    assert count == 5


def test_q01_rdd_first_action(spark_context):
    """Test first action on RDD."""
    data = [1, 2, 3, 4, 5]
    rdd = spark_context.parallelize(data)
    
    first = rdd.first()
    assert first == 1


def test_q01_rdd_take_action(spark_context):
    """Test take action on RDD."""
    data = [1, 2, 3, 4, 5]
    rdd = spark_context.parallelize(data)
    
    first_three = rdd.take(3)
    assert len(first_three) == 3
    assert first_three == [1, 2, 3]


def test_q01_rdd_chained_operations(spark_context):
    """Test chained transformations and actions."""
    data = [1, 2, 3, 4, 5, 6]
    rdd = spark_context.parallelize(data)
    
    # Chain: map -> filter -> collect
    result = rdd.map(lambda x: x ** 2) \
                .filter(lambda x: x > 10) \
                .collect()
    
    assert result == [16, 25, 36]
