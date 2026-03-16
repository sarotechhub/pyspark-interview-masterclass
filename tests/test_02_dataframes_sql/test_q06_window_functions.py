"""
Test for Q6: Window Functions Deep Dive
"""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def test_q06_row_number_ranking(spark):
    """Test row_number() window function."""
    data = [
        ("Sales", 100),
        ("Sales", 150),
        ("Marketing", 120),
        ("Marketing", 80),
        ("IT", 200)
    ]
    df = spark.createDataFrame(data, ["dept", "salary"])
    
    # Rank employees by salary within department
    window = Window.partitionBy("dept").orderBy(F.desc("salary"))
    result = df.withColumn("rank", F.row_number().over(window))
    
    collected = result.collect()
    # Sales dept: highest salary (150) should have rank 1
    sales_rank1 = [row for row in collected if row[0] == "Sales" and row[2] == 1]
    assert len(sales_rank1) == 1
    assert sales_rank1[0][1] == 150


def test_q06_rank_vs_dense_rank(spark):
    """Test rank() vs dense_rank() with ties."""
    data = [
        ("Alice", 100),
        ("Bob", 100),
        ("Carol", 90),
        ("David", 90),
        ("Eve", 80)
    ]
    df = spark.createDataFrame(data, ["name", "score"])
    
    window = Window.orderBy(F.desc("score"))
    
    # rank(): 1, 1, 3, 3, 5 (gaps on tie)
    with_rank = df.withColumn("rank", F.rank().over(window))
    ranks = sorted(with_rank.select("rank").collect())
    assert ranks[3][0] == 3  # Fourth highest has rank 3 (skip rank 2)
    
    # dense_rank(): 1, 1, 2, 2, 3 (no gaps)
    with_dense = df.withColumn("dense_rank", F.dense_rank().over(window))
    dense_ranks = sorted(with_dense.select("dense_rank").collect())
    assert dense_ranks[3][0] == 2  # Fourth highest has dense_rank 2


def test_q06_lag_lead_functions(spark):
    """Test lag() and lead() for accessing previous/next rows."""
    data = [
        ("2024-01-01", 100),
        ("2024-01-02", 110),
        ("2024-01-03", 105),
        ("2024-01-04", 120)
    ]
    df = spark.createDataFrame(data, ["date", "price"])
    
    window = Window.orderBy("date")
    
    # lag(): get previous row value
    with_lag = df.withColumn("prev_price", F.lag("price").over(window))
    result = with_lag.collect()
    
    assert result[0][2] is None      # First row has no previous
    assert result[1][2] == 100       # Second row: lag = first row's price
    assert result[2][2] == 110       # Third row: lag = second row's price
    
    # lead(): get next row value
    with_lead = df.withColumn("next_price", F.lead("price").over(window))
    result_lead = with_lead.collect()
    
    assert result_lead[0][2] == 110  # First row: lead = second row's price
    assert result_lead[3][2] is None  # Last row has no next


def test_q06_running_aggregate(spark):
    """Test running sum using window function."""
    data = [
        ("2024-01-01", 100),
        ("2024-01-02", 150),
        ("2024-01-03", 120),
        ("2024-01-04", 200)
    ]
    df = spark.createDataFrame(data, ["date", "amount"])
    
    window = Window.orderBy("date") \
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    
    result = df.withColumn("running_total", F.sum("amount").over(window))
    
    collected = result.collect()
    assert collected[0][2] == 100        # First: 100
    assert collected[1][2] == 250        # Second: 100 + 150
    assert collected[2][2] == 370        # Third: 100 + 150 + 120
    assert collected[3][2] == 570        # Fourth: 100 + 150 + 120 + 200


def test_q06_first_last_in_window(spark):
    """Test first() and last() window functions."""
    data = [
        ("A", "2024-01-01", 10),
        ("A", "2024-01-02", 20),
        ("B", "2024-01-01", 30),
        ("B", "2024-01-02", 40)
    ]
    df = spark.createDataFrame(data, ["group", "date", "value"])
    
    window = Window.partitionBy("group").orderBy("date")
    
    result = df.withColumn("first_val", F.first("value").over(window)) \
               .withColumn("last_val", F.last("value").over(window))
    
    collected = result.collect()
    # Group A: first=10, last=20
    group_a = [row for row in collected if row[0] == "A"]
    assert group_a[0][3] == 10
    assert group_a[1][4] == 20


def test_q06_ntile_percentiles(spark):
    """Test ntile() for percentile bucketing."""
    data = [(i,) for i in range(1, 11)]  # 1-10
    df = spark.createDataFrame(data, ["value"])
    
    window = Window.orderBy("value")
    
    # Divide into 4 quartiles
    quartiles = df.withColumn("quartile", F.ntile(4).over(window))
    
    collected = quartiles.collect()
    # First 2-3 values in quartile 1, next 2-3 in quartile 2, etc.
    q1_values = [row[1] for row in collected if row[1] == 1]
    q4_values = [row[1] for row in collected if row[1] == 4]
    
    assert len(q1_values) > 0
    assert len(q4_values) > 0


def test_q06_percent_rank(spark):
    """Test percent_rank() window function."""
    data = [(i,) for i in [10, 20, 20, 30, 40]]
    df = spark.createDataFrame(data, ["score"])
    
    window = Window.orderBy("score")
    
    result = df.withColumn("pct_rank", F.percent_rank().over(window))
    
    collected = result.collect()
    # percent_rank values should be between 0 and 1
    pct_ranks = [row[1] for row in collected]
    
    assert min(pct_ranks) == 0  # Minimum should be 0
    assert max(pct_ranks) == 1  # Maximum should be 1
