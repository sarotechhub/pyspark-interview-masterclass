"""
Test for Q7 (Scenario): Parse full name
"""

import pytest
from pyspark.sql import functions as F


def test_q07_split_full_name(spark):
    """Test splitting full name into parts."""
    data = [
        ("John Doe",),
        ("Alice Johnson Smith",),
        ("Bob",),
    ]
    df = spark.createDataFrame(data, ["full_name"])
    
    # Split name by space
    split_names = df.withColumn(
        "name_parts",
        F.split(F.col("full_name"), " ")
    )
    
    collected = split_names.collect()
    
    assert len(collected[0][1]) == 2  # John, Doe
    assert len(collected[1][1]) == 3  # Alice, Johnson, Smith
    assert len(collected[2][1]) == 1  # Bob


def test_q07_extract_first_last_name(spark):
    """Test extracting first and last names."""
    data = [
        ("John Doe",),
        ("Alice Johnson Smith",),
        ("Bob Wilson",),
    ]
    df = spark.createDataFrame(data, ["full_name"])
    
    # Extract first and last names
    parsed = df.withColumn(
        "first_name",
        F.split(F.col("full_name"), " ")[0]
    ).withColumn(
        "last_name",
        F.split(F.col("full_name"), " ")[
            F.size(F.split(F.col("full_name"), " ")) - 1
        ]
    )
    
    collected = parsed.collect()
    
    assert collected[0][1] == "John"
    assert collected[0][2] == "Doe"
    assert collected[1][1] == "Alice"
    assert collected[1][2] == "Smith"


def test_q07_extract_middle_initial(spark):
    """Test extracting middle initial when available."""
    data = [
        ("John Michael Doe",),
        ("Alice Smith",),
    ]
    df = spark.createDataFrame(data, ["full_name"])
    
    # Extract middle initial if exists
    parsed = df.withColumn(
        "parts",
        F.split(F.col("full_name"), " ")
    ).withColumn(
        "middle_initial",
        F.when(
            F.size(F.col("parts")) >= 3,
            F.substring(F.col("parts")[1], 1, 1)
        ).otherwise(None)
    )
    
    collected = parsed.collect()
    
    assert collected[0][2] == "M"  # John Michael Doe -> M
    assert collected[1][2] is None  # Alice Smith -> None


def test_q07_case_when_for_name_parsing(spark):
    """Test using case_when for complex name parsing."""
    data = [
        ("Dr. John Smith",),
        ("Jane Doe",),
        ("Prof. Alice Johnson",),
    ]
    df = spark.createDataFrame(data, ["full_name"])
    
    # Detect title and extract name
    parsed = df.withColumn(
        "has_title",
        F.when(
            (F.col("full_name").like("Dr.%")) | (F.col("full_name").like("Prof.%")),
            True
        ).otherwise(False)
    )
    
    with_title = parsed.filter(F.col("has_title") == True).count()
    
    assert with_title == 2
