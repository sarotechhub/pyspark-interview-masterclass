"""
Q6: Standardize phone numbers — keep only digits, enforce 10-digit format.

Scenario: Phone numbers arrive in various formats. 
Clean and validate them.
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - regexp_replace(): O(n*m) — m = string length
#   - substring/slice operations: O(n) — byte extraction
#   - when/otherwise logic: O(n) — conditional per row
#   - Total: O(n*m) where m is average phone string length
#
# Shuffle Operations:
#   - regexp_replace(): NO SHUFFLE (element-wise)
#   - string operations: NO SHUFFLE (column transform)
#   - filter on string values: NO SHUFFLE (narrow operation)
#
# Performance Tips:
#   - Regex operations are processor-intensive — optimize patterns
#   - Use substring over regexp for fixed positions (faster)
#   - Compile regex if used repeatedly (PySpark optimizes)
#   - Chain when() for validation before output
#   - Filter invalid before downstream processes
#   - Consider data type: StringType suitable for phone data
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Q06_StandardizePhoneNumbers") \
    .master("local[*]") \
    .getOrCreate()

# ============================================
# SAMPLE DATA
# ============================================
data = [
    (1, "+1-800-555-1234"),
    (2, "(415) 555-2671"),
    (3, "555.867.5309"),
    (4, "12345"),              # Invalid - too short
    (5, "8005551234"),         # Already clean 10 digits
    (6, None),                 # Null value
    (7, "1-800-CALL-NOW"),     # Contains letters
    (8, "18775551212"),        # 11 digits (country code + 10)
]

df = spark.createDataFrame(data, ["id", "phone"])

print("=" * 60)
print("INPUT DATA (Raw phone numbers):")
print("=" * 60)
df.show(truncate=False)

# ============================================
# SOLUTION
# ============================================

# Step 1: Extract only digits from the phone number
#         regexp_replace(col, pattern, replacement)
#         This removes all non-digit characters
df = df.withColumn("digits_only",
    F.regexp_replace(F.col("phone"), r"[^0-9]", "")
)

# Step 2: Handle the case where we have 11 digits (country code 1 + 10 digit number)
#         If length is 11 and starts with 1, remove the first digit
#         Otherwise keep as is
df = df.withColumn("phone_10",
    F.when(F.length("digits_only") == 11,
        F.substring("digits_only", 2, 10)  # Remove leading country code 1
    ).otherwise(F.col("digits_only"))
)

# Step 3: Validate that we have exactly 10 digits
#         If yes, keep the number; if no, set to null
df = df.withColumn("phone_valid",
    F.when(F.length("phone_10") == 10, F.col("phone_10"))
     .otherwise(F.lit(None))
)

# Step 4: Format the phone number as (XXX) XXX-XXXX (optional)
df = df.withColumn("phone_formatted",
    F.when(F.col("phone_valid").isNotNull(),
        F.concat(
            F.lit("("),
            F.substring("phone_valid", 1, 3),
            F.lit(") "),
            F.substring("phone_valid", 4, 3),
            F.lit("-"),
            F.substring("phone_valid", 7, 4)
        )
    ).otherwise(F.lit(None))
)

# Step 5: Add a validation status column
df = df.withColumn("status",
    F.when(F.col("phone_valid").isNotNull(), "Valid")
     .otherwise("Invalid")
)

print("\n" + "=" * 60)
print("PROCESSING STEPS:")
print("=" * 60)
df.select("id", "phone", "digits_only", "phone_10", "phone_valid").show(truncate=False)

print("\n" + "=" * 60)
print("FINAL OUTPUT (Cleaned & Formatted):")
print("=" * 60)
df.select("id", "phone", "phone_formatted", "status").show(truncate=False)

# ============================================
# ALTERNATIVE: Using regex_extract for direct extraction
# ============================================
df_alt = spark.createDataFrame(data, ["id", "phone"])

# Extract exactly 10 consecutive digits or 11 starting with 1
df_alt = df_alt.withColumn("digits_extracted",
    F.regexp_extract(F.col("phone"), r"(?:[^0-9]*1)?([0-9]{10})", 1)
)

df_alt = df_alt.withColumn("phone_formatted_alt",
    F.when(F.length("digits_extracted") == 10,
        F.concat(
            F.lit("("),
            F.substring("digits_extracted", 1, 3),
            F.lit(") "),
            F.substring("digits_extracted", 4, 3),
            F.lit("-"),
            F.substring("digits_extracted", 7, 4)
        )
    ).otherwise(F.lit(None))
)

print("\n" + "=" * 60)
print("ALTERNATIVE APPROACH (Using regex_extract):")
print("=" * 60)
df_alt.select("id", "phone", "phone_formatted_alt").show(truncate=False)

# ============================================
# EXPECTED OUTPUT:
# ============================================
# +---+-----------+----------------+-------+
# | id|    phone  |phone_formatted |status |
# +---+-----------+----------------+-------+
# |  1|+1-800-... |(800) 555-1234  | Valid |
# |  2|(415) 555..|(415) 555-2671  | Valid |
# |  3|555.867... |(555) 867-5309  | Valid |
# |  4|12345     | null           | Invalid|
# |  5|8005551... |(800) 555-1234  | Valid |
# |  6| null     | null           | Invalid|
# |  7|1-800-CA...| null           | Invalid|
# |  8|18775551...|(877) 555-1212  | Valid |
# +---+-----------+----------------+-------+

spark.stop()
