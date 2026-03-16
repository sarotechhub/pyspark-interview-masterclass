"""
Q16: Find customers who exist in table A but NOT in table B (anti join).

Scenario: Find customers who registered but never placed an order.
"""

# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================
# Time Complexity:
#   - join(how='left_anti'): O(n+m) — hash join with filtering
#   - n = left table rows, m = right table rows
#   - Hash table build: O(m) — create from right table
#   - Probe: O(n) — scan left table, check against hash table
#   - Total: O(n+m) — linear in combined size
#
# Shuffle Operations:
#   - join(): FULL SHUFFLE (redistribute both tables by join key)
#   - left_anti join minimizes output (no duplicates)
#
# Performance Tips:
#   - left_anti is efficient (filters without duplication)
#   - Ideally, right table (orders) smaller than left (customers)
#   - Broadcast right table if small (< spark.sql.autoBroadcastJoinThreshold)
#   - Pre-filter both tables before join to reduce data
#   - Use distinct() on keys if duplicates exist
#   - Consider: !isin() as alternative for small right table
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Q16_AntiJoin") \
    .master("local[*]") \
    .getOrCreate()

# ============================================
# SAMPLE DATA
# ============================================
# All registered customers
customers = spark.createDataFrame([
    (1, "Alice"),
    (2, "Bob"),
    (3, "Carol"),
    (4, "Dave"),
    (5, "Eve"),
], ["customer_id", "name"])

# Customers who placed orders
orders = spark.createDataFrame([
    (1, "ORD001"),
    (3, "ORD002"),
    (5, "ORD003"),
], ["customer_id", "order_id"])

print("=" * 60)
print("TABLE A: All Customers")
print("=" * 60)
customers.show()

print("=" * 60)
print("TABLE B: Customers with Orders")
print("=" * 60)
orders.show()

# ============================================
# SOLUTION
# ============================================

# Step 1: Use left_anti join
#         This returns all rows from the LEFT table (customers)
#         that have NO matching rows in the RIGHT table (orders)
# 
# Think of it as: "Give me customers MINUS customers with orders"

no_orders = customers.join(orders, on="customer_id", how="left_anti")

print("\n" + "=" * 60)
print("RESULT: Customers with NO Orders (Anti Join):")
print("=" * 60)
no_orders.show()

# ============================================
# ALTERNATIVE 1: Using NOT IN subquery
# ============================================

# Get list of customer IDs that have orders
order_customer_ids = orders.select("customer_id").distinct()

no_orders_alt1 = customers.join(
    order_customer_ids,
    on="customer_id",
    how="left_anti"
)

print("\n" + "=" * 60)
print("ALTERNATIVE 1: Using left_anti (equivalent)")
print("=" * 60)
no_orders_alt1.show()

# ============================================
# ALTERNATIVE 2: Using left join + filter null
# ============================================

# Left join and then filter where the right side is null
no_orders_alt2 = customers.join(
    orders,
    on="customer_id",
    how="left"
).filter(F.col("order_id").isNull()).select("customer_id", "name")

print("\n" + "=" * 60)
print("ALTERNATIVE 2: Using left join + filter null:")
print("=" * 60)
no_orders_alt2.show()

# ============================================
# ALTERNATIVE 3: Using NOT EXISTS pattern (SQL)
# ============================================

# Register tables for SQL
customers.createOrReplaceTempView("customers_table")
orders.createOrReplaceTempView("orders_table")

sql_result = spark.sql("""
    SELECT c.customer_id, c.name
    FROM customers_table c
    WHERE NOT EXISTS (
        SELECT 1 FROM orders_table o
        WHERE o.customer_id = c.customer_id
    )
""")

print("\n" + "=" * 60)
print("ALTERNATIVE 3: Using SQL NOT EXISTS:")
print("=" * 60)
sql_result.show()

# ============================================
# ANALYSIS: Different join types
# ============================================

print("\n" + "=" * 60)
print("COMPARISON OF DIFFERENT JOIN TYPES:")
print("=" * 60)

# Inner join - only customers with orders
inner = customers.join(orders, on="customer_id", how="inner")
print("INNER JOIN (customers WITH orders):")
inner.select("customer_id", "name").show()

# Left join - all customers, show which have orders
left = customers.join(orders, on="customer_id", how="left")
print("\nLEFT JOIN (all customers, with order info):")
left.select("customer_id", "name", "order_id").show()

# Left anti join - customers WITHOUT orders
print("\nLEFT ANTI JOIN (customers WITHOUT orders):")
no_orders.show()

# Right join - all from orders, matching customers
right = customers.join(orders, on="customer_id", how="right")
print("\nRIGHT JOIN (all orders, matching customers):")
right.show()

# Full outer join - all from both tables
full = customers.join(orders, on="customer_id", how="full")
print("\nFULL OUTER JOIN (all rows from both tables):")
full.show()

# ============================================
# EXPECTED OUTPUT:
# ============================================
# +----------+-----+
# |customer_id|name |
# +----------+-----+
# |         2 | Bob |
# |         4 | Dave|
# +----------+-----+
# 
# These customers (Bob and Dave) are in the customers table
# but have no corresponding rows in the orders table

spark.stop()
