"""
Test for Q14 (Scenario): Percentile buckets/quartiles
"""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def test_q14_ntile_quartiles(spark):
    """Test ntile for quartile bucketing."""
    data = [(i,) for i in range(1, 101)]  # 1-100
    df = spark.createDataFrame(data, ["value"])
    
    window = Window.orderBy("value")
    
    # Divide into 4 quartiles
    result = df.withColumn("quartile", F.ntile(4).over(window))
    
    q1 = result.filter(F.col("quartile") == 1).count()
    q4 = result.filter(F.col("quartile") == 4).count()
    
    # Should be roughly equal distribution
    assert q1 == 25
    assert q4 == 25


def test_q14_percent_rank(spark):
    """Test percent_rank for percentile calculation."""
    data = [(i,) for i in [10, 20, 20, 30, 40, 50]]
    df = spark.createDataFrame(data, ["score"])
    
    window = Window.orderBy("score")
    
    result = df.withColumn("pct_rank", F.percent_rank().over(window))
    
    collected = result.collect()
    
    # Check range [0, 1]
    min_pct = min(row[1] for row in collected)
    max_pct = max(row[1] for row in collected)
    
    assert min_pct == 0
    assert max_pct == 1


def test_q14_cume_dist(spark):
    """Test cume_dist for cumulative distribution."""
    data = [(i,) for i in [10, 20, 20, 30, 40, 50]]
    df = spark.createDataFrame(data, ["score"])
    
    window = Window.orderBy("score")
    
    result = df.withColumn("cume_dist", F.cume_dist().over(window))
    
    collected = result.collect()
    
    # Last row should have cume_dist = 1
    assert collected[-1][1] == 1
