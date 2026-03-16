"""
Test for Q2 (Scenario): Flatten nested JSON/struct columns
"""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType


def test_q02_flatten_nested_struct(spark):
    """Test flattening nested struct columns."""
    data = [
        ("Alice", ("NYC", "USA")),
        ("Bob", ("LA", "USA")),
    ]
    df = spark.createDataFrame(
        data,
        StructType([
            StructField("name", StringType()),
            StructField("address", StructType([
                StructField("city", StringType()),
                StructField("country", StringType()),
            ]))
        ])
    )
    
    # Flatten by extracting nested fields
    flattened = df.select(
        F.col("name"),
        F.col("address.city").alias("city"),
        F.col("address.country").alias("country")
    )
    
    assert "city" in flattened.columns
    assert "country" in flattened.columns
    
    collected = flattened.collect()
    assert collected[0][1] == "NYC"
    assert collected[0][2] == "USA"


def test_q02_flatten_array_of_structs(spark):
    """Test flattening array of struct columns."""
    from pyspark.sql.types import ArrayType
    
    data = [
        ("Order1", [("Item1", 10), ("Item2", 20)]),
        ("Order2", [("Item3", 30)]),
    ]
    
    schema = StructType([
        StructField("order_id", StringType()),
        StructField("items", ArrayType(StructType([
            StructField("item_name", StringType()),
            StructField("quantity", IntegerType()),
        ])))
    ])
    
    df = spark.createDataFrame(data, schema)
    
    # Explode array to separate rows
    exploded = df.select(
        F.col("order_id"),
        F.explode(F.col("items")).alias("item")
    ).select(
        F.col("order_id"),
        F.col("item.item_name").alias("item_name"),
        F.col("item.quantity").alias("quantity")
    )
    
    assert exploded.count() == 3
    assert "item_name" in exploded.columns


def test_q02_nested_null_handling(spark):
    """Test handling nulls in nested structures."""
    data = [
        ("Alice", ("NYC", "USA")),
        ("Bob", (None, "USA")),
        ("Carol", None),
    ]
    
    df = spark.createDataFrame(
        data,
        StructType([
            StructField("name", StringType()),
            StructField("address", StructType([
                StructField("city", StringType()),
                StructField("country", StringType()),
            ]))
        ])
    )
    
    # Extract with null safety
    result = df.select(
        F.col("name"),
        F.col("address.city").alias("city")
    )
    
    assert result.count() == 3
    assert result.filter(F.col("city").isNull()).count() == 2
