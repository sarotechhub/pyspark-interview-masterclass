"""
Q20: Join two large tables efficiently to avoid OOM (skewed join).

Scenario: One key (customer_id = 0 for guest) dominates 80% of rows.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Q20_SkewedJoin") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# ============================================
# SAMPLE DATA (Simulated skew)
# ============================================

# Create skewed orders - customer 0 (guest) has 80% of orders
from pyspark.sql.types import StructType, StructField, LongType, IntegerType

# Generate skewed data
import random
random.seed(42)

orders_data = []
# 80% guest orders (customer_id = 0)
for i in range(800):
    orders_data.append((i, 0, int(random.random() * 1000)))

# 20% regular customers
for cust_id in range(1, 101):
    for _ in range(2):
        orders_data.append((len(orders_data), cust_id, int(random.random() * 1000)))

large_orders = spark.createDataFrame(
    orders_data,
    ["order_id", "customer_id", "amount"]
)

# Small customer reference table
customers = spark.createDataFrame(
    [(i, f"Customer_{i}") for i in range(101)],
    ["customer_id", "customer_name"]
)

print("=" * 60)
print(f"ORDERS: Total {large_orders.count()} rows")
print("Customer distribution (top 5):")
print("=" * 60)
large_orders.groupBy("customer_id").count().orderBy(F.desc("count")).limit(5).show()

print("=" * 60)
print("CUSTOMERS: {} rows".format(customers.count()))
print("=" * 60)
customers.show(5)

# ============================================
# SOLUTION 1: Salt the skewed key
# ============================================

SALT = 10

# Add random salt to large table's skewed key
orders_salted = large_orders.withColumn(
    "salted_key",
    F.concat(
        F.col("customer_id"),
        F.lit("_"),
        (F.rand() * SALT).cast("int")
    )
)

# Explode the small table for all salt values
# Create array of integers 0 to SALT-1
customers_exploded = customers \
    .withColumn("salt", F.explode(F.array([F.lit(i) for i in range(SALT)]))) \
    .withColumn("salted_key", F.concat(F.col("customer_id"), F.lit("_"), F.col("salt"))) \
    .drop("salt")

# Join on salted key
result_salted = orders_salted.join(
    customers_exploded,
    on="salted_key",
    how="inner"
).drop("salted_key")

print("\n" + "=" * 60)
print("SOLUTION 1: Salted Join (distributes skew):")
print("=" * 60)
print(f"Result rows: {result_salted.count()}")
result_salted.select("order_id", "customer_id_1", "customer_name", "amount").limit(10).show()

# ============================================
# SOLUTION 2: Broadcast join for small table
# ============================================

# Use broadcast for small dimension table
result_broadcast = large_orders.join(
    F.broadcast(customers),
    on="customer_id",
    how="inner"
)

print("\n" + "=" * 60)
print("SOLUTION 2: Broadcast Join (small table on all nodes):")
print("=" * 60)
print(f"Result rows: {result_broadcast.count()}")
result_broadcast.select("order_id", "customer_id", "customer_name", "amount").limit(10).show()

# ============================================
# SOLUTION 3: Isolate skewed key and handle separately
# ============================================

# Separate skewed key (customer_id = 0)
guest_orders = large_orders.filter(F.col("customer_id") == 0)
regular_orders = large_orders.filter(F.col("customer_id") != 0)

# Join regular orders normally
regular_result = regular_orders.join(customers, on="customer_id", how="inner")

# Join guest orders
guest_customer = customers.filter(F.col("customer_id") == 0)
guest_result = guest_orders.join(guest_customer, on="customer_id", how="inner")

# Combine results
result_isolated = regular_result.union(guest_result)

print("\n" + "=" * 60)
print("SOLUTION 3: Isolate Skew (handle separately):")
print("=" * 60)
print(f"Result rows: {result_isolated.count()}")
print(f"Regular customer orders: {regular_result.count()}")
print(f"Guest orders: {guest_result.count()}")

# ============================================
# COMPARISON AND OPTIMIZATION
# ============================================

print("\n" + "=" * 60)
print("OPTIMIZATION COMPARISON:")
print("=" * 60)

# Enable adaptive query execution
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

# Show explain plans
print("\nNormal Join Plan:")
large_orders.join(customers, on="customer_id").explain()

print("\nBroadcast Join Plan:")
large_orders.join(F.broadcast(customers), on="customer_id").explain()

# ============================================
# BEST PRACTICE
# ============================================
print("\n" + "=" * 60)
print("BEST PRACTICES FOR SKEWED JOINS:")
print("=" * 60)
print("""
1. DETECT SKEW:
   - Check partition sizes: df.groupBy(spark_partition_id()).count()
   - Identify dominant keys
   
2. SOLUTIONS (in order of preference):
   - Broadcast small table (if < 10MB)
   - Salt skewed keys to redistribute data
   - Isolate skewed keys and handle separately
   - Enable adaptive query execution (Spark 3.0+)
   
3. MONITOR:
   - Use explain() to verify join type
   - Check Spark UI for shuffle sizes
   - Monitor task durations for stragglers
   
4. CONFIG:
   - spark.sql.autoBroadcastJoinThreshold (default 10MB)
   - spark.sql.adaptive.skewJoin.enabled = true
   - spark.sql.shuffle.partitions (tune for data size)
""")

# ============================================
# EXPECTED OUTPUT:
# ============================================
# ORDER DISTRIBUTION (SKEWED):
# customer_id   count
# 0             800   (80% guest orders)
# 1-100         20    (20% regular orders)
#
# After join - should have balanced partitions
# reducing task imbalance and OOM risk

spark.stop()
