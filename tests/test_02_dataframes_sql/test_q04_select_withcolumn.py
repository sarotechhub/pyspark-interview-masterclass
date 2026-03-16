"""
Test for Q4: select() vs withColumn() — DataFrame Operations
"""

import pytest
from pyspark.sql import functions as F


def test_q04_select_projection(spark):
    """Test select() for column projection."""
    data = [("Alice", 25, 50000), ("Bob", 30, 60000), ("Carol", 35, 70000)]
    df = spark.createDataFrame(data, ["name", "age", "salary"])
    
    # Select subset of columns
    selected = df.select("name", "age")
    
    assert selected.columns == ["name", "age"]
    assert selected.count() == 3
    assert len(selected.columns) == 2


def test_q04_select_with_expressions(spark):
    """Test select() with expressions."""
    data = [("Alice", 25), ("Bob", 30), ("Carol", 35)]
    df = spark.createDataFrame(data, ["name", "age"])
    
    # Select with computed expression
    result = df.select(
        F.col("name"),
        F.col("age"),
        (F.col("age") + 5).alias("age_in_5_years")
    )
    
    assert result.columns == ["name", "age", "age_in_5_years"]
    collected = result.collect()
    assert collected[0][2] == 30  # 25 + 5


def test_q04_withcolumn_add_column(spark):
    """Test withColumn() to add new columns."""
    data = [("Alice", 50000), ("Bob", 60000), ("Carol", 70000)]
    df = spark.createDataFrame(data, ["name", "salary"])
    
    # Add new column
    with_bonus = df.withColumn("bonus", F.col("salary") * 0.1)
    
    assert "bonus" in with_bonus.columns
    assert with_bonus.count() == 3
    
    collected = with_bonus.collect()
    assert collected[0][2] == 5000  # 50000 * 0.1


def test_q04_withcolumn_modify_column(spark):
    """Test withColumn() to modify existing columns."""
    data = [("Alice", 25), ("Bob", 30), ("Carol", 35)]
    df = spark.createDataFrame(data, ["name", "age"])
    
    # Modify existing column
    modified = df.withColumn("age", F.col("age") + 1)
    
    collected = modified.collect()
    assert collected[0][1] == 26  # 25 + 1
    assert collected[1][1] == 31  # 30 + 1


def test_q04_withcolumn_type_conversion(spark):
    """Test withColumn() for type conversion."""
    data = [("Alice", "25"), ("Bob", "30"), ("Carol", "35")]
    df = spark.createDataFrame(data, ["name", "age_str"])
    
    # Convert string to int
    converted = df.withColumn("age_int", F.col("age_str").cast("int"))
    
    collected = converted.collect()
    assert isinstance(converted.collect()[0][2], int)


def test_q04_select_vs_withcolumn_difference(spark):
    """Test the key difference between select() and withColumn()."""
    data = [("Alice", 25, 50000), ("Bob", 30, 60000)]
    df = spark.createDataFrame(data, ["name", "age", "salary"])
    
    # select() returns ONLY specified columns
    selected = df.select("name", F.col("salary") * 2)
    assert len(selected.columns) == 2
    
    # withColumn() keeps all existing columns and adds/modifies
    with_col = df.withColumn("doubled_salary", F.col("salary") * 2)
    assert len(with_col.columns) == 4  # name, age, salary, doubled_salary


def test_q04_chained_operations(spark):
    """Test chaining select() and withColumn()."""
    data = [("Alice", 25, 50000), ("Bob", 30, 60000), ("Carol", 35, 70000)]
    df = spark.createDataFrame(data, ["name", "age", "salary"])
    
    # Chain operations
    result = df \
        .withColumn("bonus", F.col("salary") * 0.1) \
        .select("name", "salary", "bonus")
    
    assert result.columns == ["name", "salary", "bonus"]
    assert result.count() == 3
