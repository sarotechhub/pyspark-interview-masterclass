"""
Test for Q8 (Scenario): Convert multiple date formats
"""

import pytest
from pyspark.sql import functions as F


def test_q08_coalesce_date_formats(spark):
    """Test coalescing multiple date format columns."""
    data = [
        ("2024-01-15", None, None),
        (None, "01/15/2024", None),
        (None, None, "15-Jan-2024"),
    ]
    df = spark.createDataFrame(data, ["iso_date", "us_date", "other_date"])
    
    # Try each format in order
    result = df.withColumn(
        "parsed_date",
        F.coalesce(
            F.col("iso_date"),
            F.col("us_date"),
            F.col("other_date")
        )
    )
    
    collected = result.collect()
    assert collected[0][3] == "2024-01-15"
    assert collected[1][3] == "01/15/2024"
    assert collected[2][3] == "15-Jan-2024"


def test_q08_to_date_with_patterns(spark):
    """Test converting strings to dates with patterns."""
    data = [
        ("2024-01-15",),
        ("01/15/2024",),
        ("15-Jan-2024",),
    ]
    df = spark.createDataFrame(data, ["date_str"])
    
    # Try different patterns
    result = df.withColumn(
        "parsed_date",
        F.coalesce(
            F.to_date(F.col("date_str"), "yyyy-MM-dd"),
            F.to_date(F.col("date_str"), "MM/dd/yyyy"),
            F.to_date(F.col("date_str"), "dd-MMM-yyyy")
        )
    )
    
    collected = result.collect()
    
    # All should parse to same date
    assert collected[0][1] is not None
    assert collected[1][1] is not None
    assert collected[2][1] is not None


def test_q08_null_dates_handling(spark):
    """Test handling of null dates."""
    data = [
        ("2024-01-15", "invalid"),
        (None, "2024-01-20"),
        ("invalid", None),
    ]
    df = spark.createDataFrame(data, ["date1", "date2"])
    
    result = df.withColumn(
        "valid_date",
        F.coalesce(
            F.to_date(F.col("date1"), "yyyy-MM-dd"),
            F.to_date(F.col("date2"), "yyyy-MM-dd")
        )
    )
    
    assert result.filter(F.col("valid_date").isNotNull()).count() == 2


def test_q08_date_format_detection(spark):
    """Test detecting and converting different date formats."""
    data = [
        ("2024-01-15", "ISO"),
        ("01/15/2024", "US"),
        ("15.01.2024", "EU"),
    ]
    df = spark.createDataFrame(data, ["date_str", "format"])
    
    # Convert based on format
    result = df.withColumn(
        "parsed_date",
        F.when(
            F.col("format") == "ISO",
            F.to_date(F.col("date_str"), "yyyy-MM-dd")
        ).when(
            F.col("format") == "US",
            F.to_date(F.col("date_str"), "MM/dd/yyyy")
        ).when(
            F.col("format") == "EU",
            F.to_date(F.col("date_str"), "dd.MM.yyyy")
        )
    )
    
    assert result.filter(F.col("parsed_date").isNotNull()).count() == 3
