"""
Pytest configuration and fixtures for PySpark test suite.
Provides shared SparkSession fixture for all tests.
"""

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """
    Fixture that creates a SparkSession for the entire test session.
    This is more efficient than creating/destroying session per test.
    """
    session = SparkSession.builder \
        .appName("pytest-pyspark") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.driver.host", "127.0.0.1") \
        .getOrCreate()
    
    yield session
    
    # Cleanup
    session.stop()


@pytest.fixture(scope="function")
def spark_context(spark):
    """Fixture that provides SparkContext for tests."""
    return spark.sparkContext
