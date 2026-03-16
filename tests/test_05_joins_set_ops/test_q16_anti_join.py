"""
Test for Q16 (Scenario): Anti join (A not in B)
"""

import pytest
from pyspark.sql import functions as F


def test_q16_anti_join_basic(spark):
    """Test left anti join to find records not in other table."""
    df_a = spark.createDataFrame(
        [(1, "Alice"), (2, "Bob"), (3, "Carol")],
        ["id", "name"]
    )
    df_b = spark.createDataFrame(
        [(1, "Active"), (2, "Active")],
        ["id", "status"]
    )
    
    # Find records in A that are not in B
    result = df_a.join(df_b, "id", how="left_anti")
    
    assert result.count() == 1
    assert result.collect()[0][1] == "Carol"


def test_q16_anti_join_complex(spark):
    """Test anti join with multiple join keys."""
    customers = spark.createDataFrame(
        [(1, "Alice"), (2, "Bob"), (3, "Carol"), (4, "David")],
        ["id", "name"]
    )
    purchases = spark.createDataFrame(
        [(1, 100), (2, 200)],
        ["id", "amount"]
    )
    
    # Find customers without purchases
    no_purchases = customers.join(purchases, "id", how="left_anti")
    
    assert no_purchases.count() == 2
    names = [row[1] for row in no_purchases.collect()]
    assert "Carol" in names
    assert "David" in names


def test_q16_not_in_pattern(spark):
    """Test NOT IN pattern using filter and subquery."""
    df_main = spark.createDataFrame(
        [(1, "A"), (2, "B"), (3, "C"), (4, "D")],
        ["id", "value"]
    )
    
    exclude_ids = [1, 2]
    
    # Filter out IDs that are in exclude list
    result = df_main.filter(~F.col("id").isin(exclude_ids))
    
    assert result.count() == 2
    ids = [row[0] for row in result.collect()]
    assert all(id not in exclude_ids for id in ids)
