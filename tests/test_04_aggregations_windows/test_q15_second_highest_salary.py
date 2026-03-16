"""
Test for Q15 (Scenario): Second highest salary
"""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def test_q15_second_highest_salary(spark):
    """Test finding second highest salary."""
    data = [
        ("Alice", 50000),
        ("Bob", 60000),
        ("Carol", 55000),
        ("David", 60000),  # Tie with Bob
        ("Eve", 45000),
    ]
    df = spark.createDataFrame(data, ["name", "salary"])
    
    window = Window.orderBy(F.desc("salary"))
    
    ranked = df.withColumn("rank", F.dense_rank().over(window))
    second_highest = ranked.filter(F.col("rank") == 2).collect()
    
    assert len(second_highest) > 0
    assert second_highest[0][1] == 55000  # Carol's salary


def test_q15_row_number_vs_dense_rank(spark):
    """Test difference between row_number and dense_rank for duplicates."""
    data = [
        (60000,), (60000,), (55000,), (50000,),
    ]
    df = spark.createDataFrame(data, ["salary"])
    
    window = Window.orderBy(F.desc("salary"))
    
    # Using row_number
    with_row_num = df.withColumn("row_num", F.row_number().over(window)) \
        .filter(F.col("row_num") == 2)
    
    # Using dense_rank
    with_dense = df.withColumn("dense_rank", F.dense_rank().over(window)) \
        .filter(F.col("dense_rank") == 2)
    
    # row_number should return exactly 1 row (second row = 60000)
    assert with_row_num.count() == 1
    
    # dense_rank should return 0 rows (no 2nd rank, two rows tied at rank 1)
    assert with_dense.count() == 0


def test_q15_top_n_salaries(spark):
    """Test getting top N distinct salaries."""
    data = [
        ("Alice", 50000),
        ("Bob", 60000),
        ("Carol", 60000),
        ("David", 55000),
        ("Eve", 50000),
    ]
    df = spark.createDataFrame(data, ["name", "salary"])
    
    window = Window.orderBy(F.desc("salary"))
    
    # Get distinct salary ranks
    ranked = df.withColumn("salary_rank", F.dense_rank().over(window))
    
    # Top 2 distinct salaries
    top_2_salaries = ranked.filter(F.col("salary_rank") <= 2) \
        .select("salary").distinct() \
        .count()
    
    assert top_2_salaries == 2  # 60000 and 55000
