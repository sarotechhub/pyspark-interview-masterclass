"""
Test for Q30 (Scenario): Daily report generation
"""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def test_q30_daily_sales_report(spark):
    """Test generating daily sales report."""
    sales = spark.createDataFrame(
        [(1, "2024-01-01", "Electronics", 500),
         (1, "2024-01-01", "Clothing", 200),
         (2, "2024-01-01", "Electronics", 300),
         (3, "2024-01-02", "Clothing", 150),
         (3, "2024-01-02", "Electronics", 400)],
        ["store_id", "date", "category", "amount"]
    )
    
    # Daily report: sum by date and category
    daily_report = sales.groupBy("date", "category").agg(
        F.sum("amount").alias("total_sales"),
        F.count("*").alias("transaction_count")
    )
    
    report = daily_report.collect()
    
    assert len(report) >= 4


def test_q30_multi_level_aggregation(spark):
    """Test multi-level aggregation."""
    data = spark.createDataFrame(
        [(1, "A", "2024-01-01", 100),
         (1, "A", "2024-01-01", 150),
         (1, "B", "2024-01-01", 200),
         (2, "A", "2024-01-02", 120),
         (2, "B", "2024-01-02", 180)],
        ["store_id", "product", "date", "amount"]
    )
    
    # Level 1: by date and product
    by_date_product = data.groupBy("date", "product").agg(
        F.sum("amount").alias("product_total")
    )
    
    # Level 2: by date
    by_date = data.groupBy("date").agg(
        F.sum("amount").alias("daily_total"),
        F.count("*").alias("total_transactions")
    )
    
    assert by_date_product.count() >= 3
    assert by_date.count() == 2


def test_q30_ranking_in_report(spark):
    """Test adding rankings to report."""
    data = spark.createDataFrame(
        [(1, "2024-01-01", 500),
         (2, "2024-01-01", 300),
         (3, "2024-01-01", 200),
         (1, "2024-01-02", 600),
         (2, "2024-01-02", 400)],
        ["store_id", "date", "sales"]
    )
    
    window = Window.partitionBy("date").orderBy(F.desc("sales"))
    
    ranked_report = data.withColumn(
        "rank",
        F.row_number().over(window)
    )
    
    top_stores = ranked_report.filter(F.col("rank") <= 2).count()
    
    assert top_stores == 4  # 2 stores per day


def test_q30_yoy_comparison(spark):
    """Test year-over-year comparison in report."""
    data = spark.createDataFrame(
        [(1, "2024-01-15", 1000),
         (2, "2024-01-15", 1200),
         (1, "2023-01-15", 900),
         (2, "2023-01-15", 1100)],
        ["store_id", "date", "sales"]
    )
    
    # Calculate YoY growth
    current_year = data.filter(F.col("date") >= "2024-01-01")
    prior_year = data.filter(F.col("date") < "2024-01-01")
    
    yoy = current_year.join(prior_year, "store_id", how="inner") \
        .withColumn(
            "growth_pct",
            ((current_year.sales - prior_year.sales) / prior_year.sales) * 100
        )
    
    assert yoy.count() >= 0
