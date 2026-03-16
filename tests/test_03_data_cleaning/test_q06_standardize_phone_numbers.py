"""
Test for Q6 (Scenario): Standardize phone numbers
"""

import pytest
from pyspark.sql import functions as F


def test_q06_phone_regex_extraction(spark):
    """Test extracting digits from phone numbers."""
    data = [
        ("123-456-7890",),
        ("(123) 456-7890",),
        ("123.456.7890",),
        ("1234567890",),
    ]
    df = spark.createDataFrame(data, ["phone"])
    
    # Extract only digits
    standardized = df.withColumn(
        "standardized",
        F.regexp_replace(F.col("phone"), "[^0-9]", "")
    )
    
    collected = standardized.collect()
    for row in collected:
        # All should be 10 digits
        assert len(row[1]) == 10
        assert row[1].isdigit()


def test_q06_phone_format_from_digits(spark):
    """Test formatting phone numbers to standard format."""
    data = [
        ("1234567890",),
        ("9876543210",),
    ]
    df = spark.createDataFrame(data, ["digits"])
    
    # Format as (XXX) XXX-XXXX
    formatted = df.withColumn(
        "formatted",
        F.concat(
            F.lit("("),
            F.substring(F.col("digits"), 1, 3),
            F.lit(") "),
            F.substring(F.col("digits"), 4, 3),
            F.lit("-"),
            F.substring(F.col("digits"), 7, 4)
        )
    )
    
    collected = formatted.collect()
    
    assert collected[0][1] == "(123) 456-7890"
    assert collected[1][1] == "(987) 654-3210"


def test_q06_phone_length_validation(spark):
    """Test validating phone number length."""
    data = [
        ("123-456-7890",),
        ("1234567890",),
        ("12345",),
        ("123-456",),
    ]
    df = spark.createDataFrame(data, ["phone"])
    
    # Extract digits and check length
    validated = df.withColumn(
        "digits_only",
        F.regexp_replace(F.col("phone"), "[^0-9]", "")
    ).withColumn(
        "is_valid",
        F.length(F.col("digits_only")) == 10
    )
    
    valid_count = validated.filter(F.col("is_valid") == True).count()
    assert valid_count == 2


def test_q06_phone_standardization_pipeline(spark):
    """Test full phone standardization pipeline."""
    data = [
        ("123-456-7890", "valid"),
        ("(123) 456-7890", "valid"),
        ("123456", "invalid"),
    ]
    df = spark.createDataFrame(data, ["phone", "status"])
    
    result = df.withColumn(
        "digits",
        F.regexp_replace(F.col("phone"), "[^0-9]", "")
    ).withColumn(
        "valid_length",
        F.length(F.col("digits")) == 10
    )
    
    valid_phones = result.filter(F.col("valid_length") == True).count()
    assert valid_phones == 2
