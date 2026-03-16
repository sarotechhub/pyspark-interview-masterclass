"""
Test for Q1 (Scenario): Remove duplicates based on specific columns and keep latest
"""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def test_q01_remove_duplicates_keep_latest(spark):
    """Test removing duplicates and keeping latest record."""
    data = [
        (1, "Alice", "2024-01-15"),
        (1, "Alice", "2024-03-20"),  # Latest
        (2, "Bob", "2024-02-10"),    # Latest
        (2, "Bob", "2024-01-05"),
        (3, "Carol", "2024-02-28"),
    ]
    df = spark.createDataFrame(data, ["customer_id", "name", "created_date"])
    
    window = Window.partitionBy("customer_id").orderBy(F.desc("created_date"))
    result = df.withColumn("rank", F.row_number().over(window)) \
               .filter(F.col("rank") == 1) \
               .drop("rank")
    
    assert result.count() == 3
    
    # Verify latest dates per customer
    alice_data = result.filter(F.col("customer_id") == 1).collect()[0]
    bob_data = result.filter(F.col("customer_id") == 2).collect()[0]
    
    assert alice_data[2] == "2024-03-20"  # Latest date
    assert bob_data[2] == "2024-02-10"    # Latest date


def test_q01_remove_duplicates_dense_rank(spark):
    """Test removing duplicates with ties using dense_rank."""
    data = [
        (1, "Alice", "2024-03-20"),
        (1, "Alice", "2024-03-20"),  # Same date
        (2, "Bob", "2024-02-10"),
    ]
    df = spark.createDataFrame(data, ["customer_id", "name", "created_date"])
    
    window = Window.partitionBy("customer_id").orderBy(F.desc("created_date"))
    result = df.withColumn("rank", F.dense_rank().over(window)) \
               .filter(F.col("rank") == 1) \
               .drop("rank")
    
    # dense_rank with ties should keep both records with same date
    alice_records = result.filter(F.col("customer_id") == 1)
    assert alice_records.count() == 2


def test_q01_count_duplicates_per_id(spark):
    """Test counting duplicates per customer."""
    data = [
        (1, "Alice"),
        (1, "Alice"),
        (1, "Alice"),
        (2, "Bob"),
        (2, "Bob"),
    ]
    df = spark.createDataFrame(data, ["customer_id", "name"])
    
    dup_count = df.groupBy("customer_id").count()
    
    customer_1_count = dup_count.filter(F.col("customer_id") == 1).collect()[0][1]
    customer_2_count = dup_count.filter(F.col("customer_id") == 2).collect()[0][1]
    
    assert customer_1_count == 3
    assert customer_2_count == 2
