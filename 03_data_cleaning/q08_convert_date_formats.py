"""
Q8: Convert multiple date formats in one column to a standard format.

Scenario: Date column has mixed formats from different source systems.
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - to_date() single format: O(n) — parse each row
#   - coalesce() with multiple formats: O(n*k) — k = format attempts
#   - to_timestamp(): O(n) — similar to to_date()
#   - Total: O(n*k) where k is number of date format patterns
#
# Shuffle Operations:
#   - to_date() / coalesce(): NO SHUFFLE (element-wise)
#   - date operations: NO SHUFFLE (column transform)
#   - No shuffles in date conversion
#
# Performance Tips:
#   - Order date formats by frequency (most common first)
#   - coalesce() stops at first successful parse (efficient)
#   - Try most specific patterns before generic ones
#   - Bad dates fail silently → use when() to flag issues
#   - Consider fixing source formats before ETL
#   - Cast to DateType explicitly for date operations
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Q08_ConvertDateFormats") \
    .master("local[*]") \
    .getOrCreate()

# ============================================
# SAMPLE DATA
# ============================================
data = [
    (1, "2024-03-15"),
    (2, "03/15/2024"),
    (3, "15-Mar-2024"),
    (4, "20240315"),
    (5, "bad_date"),           # Invalid
    (6, "03-15-2024"),
    (7, "2024/03/15"),
    (8, None),                 # Null value
    (9, "March 15, 2024"),
]

df = spark.createDataFrame(data, ["id", "raw_date"])

print("=" * 60)
print("INPUT DATA (Mixed date formats):")
print("=" * 60)
df.show(truncate=False)

# ============================================
# SOLUTION
# ============================================

# Step 1: Use coalesce() with multiple to_date() calls
#         Each to_date() tries a different format
#         The first one that succeeds will be used
result = df.withColumn("std_date",
    F.coalesce(
        # Try standard ISO format first (YYYY-MM-DD)
        F.to_date("raw_date", "yyyy-MM-dd"),
        # Try US format (MM/DD/YYYY)
        F.to_date("raw_date", "MM/dd/yyyy"),
        # Try European format with month name (DD-MMM-YYYY)
        F.to_date("raw_date", "dd-MMM-yyyy"),
        # Try compact format (YYYYMMDD)
        F.to_date("raw_date", "yyyyMMdd"),
        # Try alternative US format (MM-DD-YYYY)
        F.to_date("raw_date", "MM-dd-yyyy"),
        # Try with slashes (YYYY/MM/DD)
        F.to_date("raw_date", "yyyy/MM/dd"),
        # Try full month name format
        F.to_date("raw_date", "MMMM dd, yyyy")
    )
)

# Step 2: Add a validation column to check if date was successfully parsed
result = result.withColumn("is_valid",
    F.col("std_date").isNotNull()
)

# Step 3: Add a status column with helpful info
result = result.withColumn("status",
    F.when(F.col("std_date").isNotNull(), "Valid")
     .otherwise("Invalid")
)

print("\n" + "=" * 60)
print("OUTPUT DATA (Standardized to date):")
print("=" * 60)
result.select("id", "raw_date", "std_date", "status").show(truncate=False)

# ============================================
# ALTERNATIVE: Using a try-catch function
# ============================================
# Define a UDF to try multiple formats

from pyspark.sql.types import DateType

def parse_date(date_str):
    """
    Try to parse a date string using multiple formats.
    Returns a date object or None if parsing fails.
    """
    from datetime import datetime
    
    if date_str is None:
        return None
    
    # List of date formats to try
    formats = [
        "%Y-%m-%d",      # 2024-03-15
        "%m/%d/%Y",      # 03/15/2024
        "%d-%b-%Y",      # 15-Mar-2024
        "%Y%m%d",        # 20240315
        "%m-%d-%Y",      # 03-15-2024
        "%Y/%m/%d",      # 2024/03/15
        "%B %d, %Y",     # March 15, 2024
        "%d/%m/%Y",      # 15/03/2024
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    
    # If no format matches, return None
    return None

# Register the UDF
parse_date_udf = F.udf(parse_date, DateType())

result_udf = spark.createDataFrame(data, ["id", "raw_date"])
result_udf = result_udf.withColumn("std_date", parse_date_udf("raw_date"))

print("\n" + "=" * 60)
print("ALTERNATIVE OUTPUT (Using Python UDF):")
print("=" * 60)
result_udf.select("id", "raw_date", "std_date").show(truncate=False)

# ============================================
# FORMATTING THE OUTPUT DATE
# ============================================
# Convert to different output formats if needed

result_formatted = result.withColumn("std_date_iso",
    F.date_format(F.col("std_date"), "yyyy-MM-dd")
)

result_formatted = result_formatted.withColumn("std_date_us",
    F.date_format(F.col("std_date"), "MM/dd/yyyy")
)

result_formatted = result_formatted.withColumn("std_date_eur",
    F.date_format(F.col("std_date"), "dd-MM-yyyy")
)

print("\n" + "=" * 60)
print("OUTPUT WITH MULTIPLE FORMATS:")
print("=" * 60)
result_formatted.select(
    "id", 
    "raw_date", 
    "std_date_iso", 
    "std_date_us", 
    "std_date_eur"
).show(truncate=False)

# ============================================
# EXPECTED OUTPUT:
# ============================================
# +---+----------+----------+-------+
# | id| raw_date | std_date | status|
# +---+----------+----------+-------+
# |  1|2024-03-15|2024-03-15| Valid |
# |  2|03/15/2024|2024-03-15| Valid |
# |  3|15-Mar-2024|2024-03-15| Valid |
# |  4|20240315 |2024-03-15| Valid |
# |  5| bad_date | null     | Invalid|
# |  6|03-15-2024|2024-03-15| Valid |
# |  7|2024/03/15|2024-03-15| Valid |
# |  8| null     | null     | Invalid|
# |  9|March 15...|2024-03-15| Valid |
# +---+----------+----------+-------+

spark.stop()
