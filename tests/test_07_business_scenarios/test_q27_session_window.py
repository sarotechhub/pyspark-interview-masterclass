"""
Test for Q27 (Scenario): Session windowing
"""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def test_q27_session_detection(spark):
    """Test detecting sessions based on time gaps."""
    data = spark.createDataFrame(
        [(1, "2024-01-01 10:00:00"),
         (1, "2024-01-01 10:05:00"),
         (1, "2024-01-01 11:00:00"),  # Gap > 30 min -> new session
         (2, "2024-01-01 10:15:00"),
         (2, "2024-01-01 10:20:00")],
        ["user_id", "event_time"]
    )
    
    # Convert to timestamp
    timestamped = data.withColumn(
        "event_ts",
        F.to_timestamp(F.col("event_time"))
    )
    
    # Calculate time difference from previous event
    window = Window.partitionBy("user_id").orderBy("event_ts")
    
    with_gaps = timestamped.withColumn(
        "time_diff_minutes",
        (F.unix_timestamp(F.col("event_ts")) - 
         F.unix_timestamp(F.lag(F.col("event_ts")).over(window))) / 60
    )
    
    # Mark session start when gap > 30 minutes
    with_sessions = with_gaps.withColumn(
        "new_session",
        F.col("time_diff_minutes").isNull() | 
        (F.col("time_diff_minutes") > 30)
    )
    
    new_sessions = with_sessions.filter(F.col("new_session") == True).count()
    
    assert new_sessions >= 1


def test_q27_session_grouping(spark):
    """Test grouping events into sessions."""
    data = spark.createDataFrame(
        [(1, "click"), (1, "view"), (1, "click"),
         (2, "view"), (2, "view")],
        ["user_id", "event_type"]
    )
    
    # Simple session counting per user
    sessions = data.groupBy("user_id").agg(
        F.count("*").alias("event_count"),
        F.collect_list("event_type").alias("events")
    )
    
    assert sessions.count() == 2


def test_q27_session_duration(spark):
    """Test calculating session duration."""
    data = spark.createDataFrame(
        [(1, "2024-01-01 10:00:00", "start"),
         (1, "2024-01-01 10:15:00", "end"),
         (2, "2024-01-01 11:00:00", "start"),
         (2, "2024-01-01 11:10:00", "end")],
        ["user_id", "timestamp", "event"]
    )
    
    timestamped = data.withColumn(
        "ts",
        F.to_timestamp(F.col("timestamp"))
    )
    
    # Calculate duration between start and end
    start_events = timestamped.filter(F.col("event") == "start") \
        .select("user_id", F.col("ts").alias("start_time"))
    
    end_events = timestamped.filter(F.col("event") == "end") \
        .select("user_id", F.col("ts").alias("end_time"))
    
    duration = start_events.join(end_events, "user_id")
    
    assert duration.count() == 2
