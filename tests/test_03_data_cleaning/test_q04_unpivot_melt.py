"""
Test for Q4 (Scenario): Unpivot (melt) columns
"""

import pytest
from pyspark.sql import functions as F


def test_q04_unpivot_basic(spark):
    """Test basic unpivot transformation."""
    data = [
        ("Alice", 100, 120),
        ("Bob", 110, 130),
    ]
    df = spark.createDataFrame(data, ["name", "Q1", "Q2"])
    
    # Unpivot: columns become rows
    unpivoted = df.select(
        F.col("name"),
        F.lit("Q1").alias("quarter"),
        F.col("Q1").alias("amount")
    ).union(
        df.select(
            F.col("name"),
            F.lit("Q2").alias("quarter"),
            F.col("Q2").alias("amount")
        )
    )
    
    assert unpivoted.count() == 4  # 2 rows * 2 quarters
    assert "quarter" in unpivoted.columns
    assert "amount" in unpivoted.columns


def test_q04_unpivot_with_stack(spark):
    """Test unpivot using stack function."""
    data = [
        ("Alice", 100, 120, 110),
        ("Bob", 110, 130, 125),
    ]
    df = spark.createDataFrame(data, ["name", "Q1", "Q2", "Q3"])
    
    # Using stack: more concise way to unpivot
    unpivoted = df.select(
        F.col("name"),
        F.expr("stack(3, 'Q1', Q1, 'Q2', Q2, 'Q3', Q3) as (quarter, amount)")
    ).select("name", "quarter", "amount")
    
    assert unpivoted.count() == 6  # 2 rows * 3 quarters
    
    q1_records = unpivoted.filter(F.col("quarter") == "Q1")
    assert q1_records.count() == 2


def test_q04_unpivot_with_multiple_value_columns(spark):
    """Test unpivot with multiple value columns."""
    data = [
        ("Product1", 100, 5, 120, 6),
        ("Product2", 110, 4, 130, 7),
    ]
    df = spark.createDataFrame(
        data,
        ["product", "Q1_sales", "Q1_units", "Q2_sales", "Q2_units"]
    )
    
    # Create month and values from quarters
    q1 = df.select(
        F.col("product"),
        F.lit("Q1").alias("quarter"),
        F.col("Q1_sales").alias("sales"),
        F.col("Q1_units").alias("units")
    )
    
    q2 = df.select(
        F.col("product"),
        F.lit("Q2").alias("quarter"),
        F.col("Q2_sales").alias("sales"),
        F.col("Q2_units").alias("units")
    )
    
    unpivoted = q1.union(q2)
    
    assert unpivoted.count() == 4
    assert "quarter" in unpivoted.columns
    assert "sales" in unpivoted.columns
    assert "units" in unpivoted.columns
