"""
Test for Q3 (Scenario): Pivot table transformation
"""

import pytest
from pyspark.sql import functions as F


def test_q03_pivot_basic(spark):
    """Test basic pivot transformation."""
    data = [
        ("Q1", "Sales", 100),
        ("Q1", "Marketing", 50),
        ("Q2", "Sales", 150),
        ("Q2", "Marketing", 80),
    ]
    df = spark.createDataFrame(data, ["quarter", "department", "budget"])
    
    # Pivot: rows become columns
    pivoted = df.pivot("department").agg(F.sum("budget"))
    
    assert "Marketing" in pivoted.columns
    assert "Sales" in pivoted.columns
    
    collected = pivoted.collect()
    q1_row = [row for row in collected if row[0] == "Q1"][0]
    
    # Q1: Marketing=50, Sales=100
    assert q1_row[1] == 50 or q1_row[2] == 50  # Marketing


def test_q03_pivot_with_aggregation(spark):
    """Test pivot with different aggregations."""
    data = [
        ("A", "X", 10),
        ("A", "X", 20),
        ("A", "Y", 30),
        ("B", "X", 40),
        ("B", "Y", 50),
    ]
    df = spark.createDataFrame(data, ["group", "category", "value"])
    
    # Pivot with sum
    pivoted_sum = df.pivot("category").agg(F.sum("value"))
    
    collected = pivoted_sum.collect()
    assert pivoted_sum.count() == 2  # 2 groups


def test_q03_pivot_multiple_aggregations(spark):
    """Test pivot with multiple aggregation functions."""
    data = [
        ("Q1", "Sales", 100),
        ("Q1", "Sales", 150),
        ("Q1", "Marketing", 50),
        ("Q2", "Sales", 120),
        ("Q2", "Marketing", 80),
    ]
    df = spark.createDataFrame(data, ["quarter", "department", "amount"])
    
    # Pivot with multiple aggregations
    pivoted = df.pivot("department").agg(
        F.sum("amount").alias("total"),
        F.count("amount").alias("count")
    )
    
    assert "Marketing_total" in pivoted.columns or "Marketing" in pivoted.columns
    assert pivoted.count() >= 2
