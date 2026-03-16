"""
Test for Q10 (Scenario): Top N per category
"""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def test_q10_top_n_per_group(spark):
    """Test getting top N records per group."""
    data = [
        ("Sales", "Alice", 50000),
        ("Sales", "Bob", 60000),
        ("Sales", "Carol", 45000),
        ("Marketing", "David", 55000),
        ("Marketing", "Eve", 52000),
    ]
    df = spark.createDataFrame(data, ["dept", "name", "salary"])
    
    window = Window.partitionBy("dept").orderBy(F.desc("salary"))
    
    ranked = df.withColumn("rank", F.dense_rank().over(window))
    top_2 = ranked.filter(F.col("rank") <= 2)
    
    sales_top = top_2.filter(F.col("dept") == "Sales").count()
    marketing_top = top_2.filter(F.col("dept") == "Marketing").count()
    
    assert sales_top == 2
    assert marketing_top == 2


def test_q10_top_1_per_category(spark):
    """Test getting single top record per category."""
    data = [
        ("A", 10), ("A", 20), ("A", 15),
        ("B", 30), ("B", 25),
        ("C", 5),
    ]
    df = spark.createDataFrame(data, ["category", "value"])
    
    window = Window.partitionBy("category").orderBy(F.desc("value"))
    
    top_per_cat = df.withColumn("rank", F.row_number().over(window)) \
        .filter(F.col("rank") == 1)
    
    assert top_per_cat.count() == 3  # One per category
    
    # Verify max values
    a_value = top_per_cat.filter(F.col("category") == "A").collect()[0][1]
    b_value = top_per_cat.filter(F.col("category") == "B").collect()[0][1]
    
    assert a_value == 20
    assert b_value == 30
