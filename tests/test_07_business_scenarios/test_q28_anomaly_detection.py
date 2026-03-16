"""
Test for Q28 (Scenario): Anomaly detection
"""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def test_q28_zscore_anomaly_detection(spark):
    """Test Z-score based anomaly detection."""
    # Normal data: mostly 100 with some anomalies
    data = spark.createDataFrame(
        [(i, 100 + (i % 5)) for i in range(100)] +  # Normal
        [(100, 999), (101, -999)],  # Anomalies
        ["id", "value"]
    )
    
    # Calculate mean and stddev
    stats = data.agg(
        F.mean("value").alias("mean_val"),
        F.stddev("value").alias("stddev_val")
    )
    
    stats_row = stats.collect()[0]
    mean_val = stats_row[0]
    stddev_val = stats_row[1]
    
    # Calculate Z-scores
    result = data.withColumn(
        "z_score",
        F.abs((F.col("value") - mean_val) / stddev_val)
    ).withColumn(
        "is_anomaly",
        F.col("z_score") > 3  # Threshold of 3 standard deviations
    )
    
    anomalies = result.filter(F.col("is_anomaly") == True).count()
    
    assert anomalies == 2


def test_q28_threshold_based_detection(spark):
    """Test simple threshold-based anomaly detection."""
    data = spark.createDataFrame(
        [(i, i * 10) for i in range(1, 11)] +  # Normal sequence
        [(11, 999)],  # Anomaly
        ["id", "value"]
    )
    
    # Set threshold as 95th percentile
    percentile_95 = data.approxQuantile("value", [0.95], 0.01)[0]
    
    anomalies = data.filter(F.col("value") > percentile_95).count()
    
    assert anomalies >= 1


def test_q28_detect_trends(spark):
    """Test detecting anomalous trends."""
    data = spark.createDataFrame(
        [(i, 100 + (i % 10)) for i in range(1, 21)] +  # Stable
        [(21, 150), (22, 200), (23, 250)],  # Sharp increase
        ["day", "value"]
    )
    
    window = Window.orderBy("day")
    
    result = data.withColumn(
        "prev_value",
        F.lag("value").over(window)
    ).withColumn(
        "pct_change",
        F.when(
            F.col("prev_value").isNotNull(),
            ((F.col("value") - F.col("prev_value")) / F.col("prev_value")) * 100
        )
    ).withColumn(
        "anomaly",
        F.col("pct_change") > 30  # More than 30% increase
    )
    
    anomalies = result.filter(F.col("anomaly") == True).count()
    
    assert anomalies >= 1
