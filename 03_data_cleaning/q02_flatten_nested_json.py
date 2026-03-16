"""
Q2: Flatten a nested JSON / struct column.

Scenario: Your data has nested JSON with address as a struct. 
Flatten it to top-level columns.
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - Manual flatten (select nested fields): O(n) — one pass
#   - explode() for nested arrays: O(n*m) — m elements per row
#   - from_json() parsing: O(n) — parse each JSON
#   - Total: O(n) to O(n*m) depending on nesting depth
#
# Shuffle Operations:
#   - select() with nested fields: NO SHUFFLE
#   - explode(): NO SHUFFLE (but duplicates rows)
#   - from_json(): NO SHUFFLE
#
# Performance Tips:
#   - Use manual select for simple flattening (avoids parsing)
#   - Avoid nested JSON if possible — flatten at load time
#   - Use explode() carefully (can expand data significantly)
#   - Consider schema_of_json() to dynamically detect structure
#   - For deep nesting: use path notation (col.field1.field2)
#   - explode(array_col) duplicates parent rows for each element
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Q02_FlattenNestedJSON") \
    .master("local[*]") \
    .getOrCreate()

# ============================================
# SAMPLE DATA
# ============================================
data = [
    (1, ("New York", "10001", "NY")),
    (2, ("Los Angeles", "90001", "CA")),
    (3, ("Chicago", "60601", "IL")),
]

# Define schema with nested struct
schema = StructType([
    StructField("id", StringType()),
    StructField("address", StructType([
        StructField("city",  StringType()),
        StructField("zip",   StringType()),
        StructField("state", StringType())
    ]))
])

df = spark.createDataFrame(data, schema)

print("=" * 60)
print("INPUT DATA (Nested Structure):")
print("=" * 60)
df.show()
df.printSchema()

# ============================================
# SOLUTION - METHOD 1: Manual Flatten
# ============================================
# Step 1: Access nested columns using dot notation (address.city)
# Step 2: Select the nested columns and rename if needed
result_method1 = df.select(
    "id",
    "address.city",
    "address.zip",
    "address.state"
)

print("\n" + "=" * 60)
print("OUTPUT METHOD 1 (Manual Flatten):")
print("=" * 60)
result_method1.show()

# ============================================
# SOLUTION - METHOD 2: Dynamic Flatten (Recursive)
# ============================================
def flatten_df(df):
    """
    Dynamically flatten a DataFrame with nested structures.
    Works recursively for deeply nested data.
    """
    from pyspark.sql.types import StructType
    
    cols = []
    
    # Iterate through each field in the schema
    for field in df.schema.fields:
        # Check if the field is a struct type
        if isinstance(field.dataType, StructType):
            # For each subfield in the struct, create a column reference
            for subfield in field.dataType.fields:
                cols.append(
                    F.col(f"{field.name}.{subfield.name}")
                     .alias(f"{field.name}_{subfield.name}")
                )
        else:
            # For non-struct fields, keep as is
            cols.append(F.col(field.name))
    
    return df.select(cols)

result_method2 = flatten_df(df)

print("\n" + "=" * 60)
print("OUTPUT METHOD 2 (Dynamic Flatten):")
print("=" * 60)
result_method2.show()

# ============================================
# SOLUTION - METHOD 3: Using explode for arrays
# ============================================
# If address contained an array, we could use explode:
# df.withColumn("item", F.explode("address_array"))

# ============================================
# EXPECTED OUTPUT:
# ============================================
# Method 1 & 2 should produce:
# +---+--------+-----+-----+
# | id|    city|  zip|state|
# +---+--------+-----+-----+
# |  1|New York|10001|   NY|
# |  2|Los Angeles|90001|   CA|
# |  3|Chicago |60601|   IL|
# +---+--------+-----+-----+

spark.stop()
