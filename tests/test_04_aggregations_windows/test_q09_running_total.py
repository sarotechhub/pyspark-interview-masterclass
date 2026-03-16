"""
Test for Q9 (Scenario): Running total (cumulative sum)
"""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def test_q09_running_total_basic(spark):
    """Test calculating running total per customer."""
    data = [
        (1, "2024-01-01", 100),
        (1, "2024-01-05", 200),
        (1, "2024-01-10", 150),
        (2, "2024-01-02", 300),
        (2, "2024-01-08", 100),
    ]
    df = spark.createDataFrame(data, ["customer_id", "order_date", "amount"])
    
    window = Window.partitionBy("customer_id") \
        .orderBy("order_date") \
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    
    result = df.withColumn("running_total", F.sum("amount").over(window))
    
    collected = result.collect()
    
    # Customer 1: 100, 300 (100+200), 450 (100+200+150)
    assert collected[0][3] == 100
    assert collected[1][3] == 300
    assert collected[2][3] == 450


def test_q09_running_total_multiple_groups(spark):
    """Test running total with multiple customer groups."""
    data = [
        (1, 100), (1, 200), (1, 150),
        (2, 300), (2, 100),
    ]
    df = spark.createDataFrame(data, ["customer_id", "amount"])
    
    window = Window.partitionBy("customer_id") \
        .orderBy(F.desc("amount")) \
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    
    result = df.withColumn("running_total", F.sum("amount").over(window))
    
    # Each group should have separate running totals
    customer_1_totals = result.filter(F.col("customer_id") == 1) \
        .select("running_total").collect()
    customer_2_totals = result.filter(F.col("customer_id") == 2) \
        .select("running_total").collect()
    
    assert len(customer_1_totals) == 3
    assert len(customer_2_totals) == 2


def test_q09_running_count(spark):
    """Test running count (number of rows accumulated)."""
    data = [(i, f"Day{i}") for i in range(1, 6)]
    df = spark.createDataFrame(data, ["day", "event"])
    
    window = Window.orderBy("day") \
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    
    result = df.withColumn("running_count", F.count("*").over(window))
    
    collected = result.collect()
    
    assert collected[0][2] == 1
    assert collected[1][2] == 2
    assert collected[4][2] == 5  # Last row
