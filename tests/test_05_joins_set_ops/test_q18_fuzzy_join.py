"""
Test for Q18 (Scenario): Fuzzy join (name matching with Levenshtein distance)
"""

import pytest
from pyspark.sql import functions as F


def test_q18_levenshtein_distance(spark):
    """Test calculating Levenshtein distance between strings."""
    data = [("Alice",), ("Alise",), ("Bob",), ("Boob",)]
    df = spark.createDataFrame(data, ["name"])
    
    # Calculate distance from reference string "Alice"
    result = df.withColumn(
        "distance",
        F.levenshtein(F.col("name"), F.lit("Alice"))
    )
    
    collected = result.collect()
    
    assert collected[0][1] == 0  # "Alice" vs "Alice" = 0
    assert collected[1][1] == 1  # "Alise" vs "Alice" = 1 (one substitution)
    assert collected[2][1] > 1   # "Bob" is different


def test_q18_fuzzy_match_threshold(spark):
    """Test fuzzy matching with similarity threshold."""
    ref_names = spark.createDataFrame([("John Smith",)], ["ref_name"])
    data = spark.createDataFrame(
        [("John Smith",), ("Jon Smith",), ("Jane Smith",)],
        ["name"]
    )
    
    # Cross join to compare all names
    compared = data.crossJoin(ref_names).withColumn(
        "distance",
        F.levenshtein(F.col("name"), F.col("ref_name"))
    )
    
    # Find matches with distance <= 2
    matches = compared.filter(F.col("distance") <= 2)
    
    assert matches.count() >= 1


def test_q18_soundex_similarity(spark):
    """Test soundex for phonetic similarity."""
    data = [("Smith",), ("Smythe",), ("Johnson",)]
    df = spark.createDataFrame(data, ["name"])
    
    # Calculate soundex codes
    result = df.withColumn(
        "soundex",
        F.soundex(F.col("name"))
    )
    
    smith_code = result.filter(F.col("name") == "Smith").collect()[0][1]
    smythe_code = result.filter(F.col("name") == "Smythe").collect()[0][1]
    
    # Smith and Smythe should have same soundex (phonetically similar)
    assert smith_code == smythe_code
