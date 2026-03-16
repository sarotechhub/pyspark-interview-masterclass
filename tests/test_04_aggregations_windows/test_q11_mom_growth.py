"""
Test for Q11 (Scenario): Month-over-month growth
"""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def test_q11_mom_growth_calculation(spark):
    """Test calculating month-over-month growth percentage."""
    data = [
        ("2024-01-01", 1000),
        ("2024-02-01", 1200),  # 20% growth
        ("2024-03-01", 1100),  # -8.33% growth
    ]
    df = spark.createDataFrame(data, ["date", "revenue"])
    
    window = Window.orderBy("date")
    
    with_mom = df.withColumn(
        "prev_revenue",
        F.lag("revenue").over(window)
    ).withColumn(
        "mom_growth_pct",
        F.round(
            ((F.col("revenue") - F.col("prev_revenue")) / F.col("prev_revenue")) * 100,
            2
        )
    )
    
    collected = with_mom.collect()
    
    assert collected[0][3] is None  # First month no previous
    assert abs(collected[1][3] - 20.0) < 0.1  # 20% growth
    assert collected[2][3] < 0  # Negative growth


def test_q11_mom_with_reference_date(spark):
    """Test MoM growth with date calculation."""
    data = [
        ("2024-01-31", 1000),
        ("2024-02-28", 1100),
        ("2024-03-31", 1200),
    ]
    df = spark.createDataFrame(data, ["month_end", "sales"])
    
    window = Window.orderBy("month_end")
    
    result = df.withColumn(
        "prev_sales",
        F.lag("sales").over(window)
    ).withColumn(
        "growth_pct",
        F.when(
            F.col("prev_sales").isNotNull(),
            ((F.col("sales") - F.col("prev_sales")) / F.col("prev_sales")) * 100
        )
    )
    
    growth_values = result.filter(F.col("growth_pct").isNotNull()) \
        .select("growth_pct").collect()
    
    assert len(growth_values) == 2
    assert all(val[0] > 0 for val in growth_values)  # All positive growth
