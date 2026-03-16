"""
Test for Q2: Transformations vs Actions — Lazy vs Eager Execution
"""

import pytest
from pyspark.sql import functions as F


def test_q02_transformations_are_lazy(spark_context):
    """Test that transformations don't execute immediately."""
    data = [1, 2, 3, 4, 5]
    rdd = spark_context.parallelize(data)
    
    # These transformations should not execute yet
    squared = rdd.map(lambda x: x ** 2)
    even = squared.filter(lambda x: x % 2 == 0)
    
    # Verify RDD objects are created but not computed
    assert squared is not None
    assert even is not None
    
    # Action triggers execution
    result = even.collect()
    assert result == [4, 16]


def test_q02_multiple_actions(spark_context):
    """Test multiple actions on same RDD."""
    data = [1, 2, 3, 4, 5]
    rdd = spark_context.parallelize(data)
    
    squared = rdd.map(lambda x: x ** 2)
    
    # Multiple actions on same RDD
    first = squared.first()
    count = squared.count()
    collect = squared.collect()
    
    assert first == 1
    assert count == 5
    assert collect == [1, 4, 9, 16, 25]


def test_q02_rdd_reduce_action(spark_context):
    """Test reduce action (aggregation)."""
    data = [1, 2, 3, 4, 5]
    rdd = spark_context.parallelize(data)
    
    # Reduce: sum all elements
    total = rdd.reduce(lambda a, b: a + b)
    assert total == 15


def test_q02_rdd_union_transformation(spark_context):
    """Test union transformation of RDDs."""
    rdd1 = spark_context.parallelize([1, 2, 3])
    rdd2 = spark_context.parallelize([4, 5, 6])
    
    # Union combines RDDs
    combined = rdd1.union(rdd2)
    result = combined.collect()
    
    assert len(result) == 6
    assert result == [1, 2, 3, 4, 5, 6]


def test_q02_rdd_distinct_transformation(spark_context):
    """Test distinct transformation."""
    data = [1, 2, 2, 3, 3, 3, 4]
    rdd = spark_context.parallelize(data)
    
    # Remove duplicates
    distinct = rdd.distinct()
    result = sorted(distinct.collect())
    
    assert result == [1, 2, 3, 4]


def test_q02_rdd_groupbykey_transformation(spark_context):
    """Test groupByKey transformation for key-value RDDs."""
    data = [("a", 1), ("a", 2), ("b", 3), ("b", 4), ("a", 5)]
    rdd = spark_context.parallelize(data)
    
    # Group by key
    grouped = rdd.groupByKey().mapValues(lambda x: list(x)).collect()
    
    # Convert to dict for easier testing
    result_dict = dict(grouped)
    
    assert set(result_dict["a"]) == {1, 2, 5}
    assert set(result_dict["b"]) == {3, 4}


def test_q02_rdd_sortbykey_transformation(spark_context):
    """Test sortByKey transformation."""
    data = [("b", 2), ("a", 1), ("c", 3), ("a", 4)]
    rdd = spark_context.parallelize(data)
    
    # Sort by key
    sorted_rdd = rdd.sortByKey()
    result = sorted_rdd.collect()
    
    # First two should be "a", then "b", then "c"
    assert result[0][0] == "a"
    assert result[1][0] == "a"
    assert result[2][0] == "b"
    assert result[3][0] == "c"
