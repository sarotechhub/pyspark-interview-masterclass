"""
Test for Q26 (Scenario): Churn detection
"""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def test_q26_identify_inactive_customers(spark):
    """Test identifying inactive customers."""
    data = spark.createDataFrame(
        [(1, "Alice", "2024-01-15"),
         (2, "Bob", "2024-02-10"),
         (3, "Carol", "2023-01-20"),  # Old date -> inactive
         (4, "David", "2024-02-28")],
        ["customer_id", "name", "last_purchase_date"]
    )
    
    # Mark as churned if last purchase > 1 year ago
    result = data.withColumn(
        "is_churned",
        F.datediff(F.lit("2024-03-01"), F.col("last_purchase_date")) > 365
    )
    
    churned = result.filter(F.col("is_churned") == True).count()
    
    assert churned >= 1


def test_q26_churn_by_days_inactive(spark):
    """Test calculating days since last activity."""
    data = spark.createDataFrame(
        [(1, "2024-02-01"), (2, "2024-01-01"), (3, "2023-12-01")],
        ["customer_id", "last_active_date"]
    )
    
    reference_date = F.lit("2024-03-01")
    
    result = data.withColumn(
        "days_inactive",
        F.datediff(reference_date, F.col("last_active_date"))
    ).withColumn(
        "churn_risk",
        F.case_when(
            F.col("days_inactive") < 30, "Low"
        ).when(
            F.col("days_inactive") < 90, "Medium"
        ).otherwise("High")
    )
    
    high_risk = result.filter(F.col("churn_risk") == "High").count()
    
    assert high_risk >= 1


def test_q26_churn_cohort_analysis(spark):
    """Test analyzing churn by cohort."""
    data = spark.createDataFrame(
        [(1, "2024-01", "Active"),
         (2, "2024-01", "Inactive"),
         (3, "2024-02", "Active"),
         (4, "2024-02", "Inactive"),
         (5, "2024-02", "Inactive")],
        ["customer_id", "cohort", "status"]
    )
    
    churn_stats = data.groupBy("cohort").agg(
        F.countIf(F.col("status") == "Inactive").alias("churned_count"),
        F.count("*").alias("total_count")
    ).withColumn(
        "churn_rate",
        F.round((F.col("churned_count") / F.col("total_count")) * 100, 2)
    )
    
    assert churn_stats.count() == 2
