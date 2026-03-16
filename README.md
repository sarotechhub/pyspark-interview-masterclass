# 🔥 PySpark Interview Preparation Guide

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PySpark](https://img.shields.io/badge/PySpark-3.5.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **Complete working examples** — 30 Theory Questions + 30 Scenario-Based Coding Questions with ready-to-run Python files

A comprehensive, hands-on PySpark interview preparation repository with practical examples, detailed comments, and expected outputs for all coding scenarios.

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
- [Question Index](#question-index)
- [Key Concepts](#key-concepts)

---

## 🚀 Quick Start

```bash
# Clone this repository
git clone https://github.com/yourusername/pyspark-interview-prep.git
cd pyspark-interview-prep

# Install dependencies
pip install -r requirements.txt

# Run a sample script
python 03_data_cleaning/q01_remove_duplicates.py
```

---

## 📁 Project Structure

**Total Files: 40** (6 BONUS + 30 Original Scenarios + 4 Configuration)

### BONUS Files — Extra Learning Resources (Not in Original 30)
```
├── 01_core_concepts/                  # BONUS: RDD & core Spark fundamentals
│   ├── q01_rdd_basics.py              # (BONUS) RDD fundamentals & operations
│   ├── q02_transformations_actions.py # (BONUS) Lazy vs eager execution
│   └── q03_sparksession.py            # (BONUS) SparkSession configuration
│
├── 02_dataframes_sql/                 # BONUS: DataFrame & SQL operations
│   ├── q04_select_withcolumn.py       # (BONUS) Select vs withColumn
│   ├── q05_null_handling.py           # (BONUS) Null value handling
│   └── q06_window_functions.py        # (BONUS) Window functions deep dive
```

### Original Interview Scenarios (Q1-Q30)
```
├── 03_data_cleaning/                  # SCENARIO Q1-Q8: Data cleaning & transformation
│   ├── q01_remove_duplicates.py       # Scenario Q1: Window + row_number
│   ├── q02_flatten_nested_json.py     # Scenario Q2: Struct flattening
│   ├── q03_pivot_table.py             # Scenario Q3: Group + pivot
│   ├── q04_unpivot_melt.py            # Scenario Q4: Stack + union
│   ├── q05_forward_fill.py            # Scenario Q5: Window fill with last()
│   ├── q06_standardize_phone_numbers.py # Scenario Q6: Regex + validation
│   ├── q07_parse_full_name.py         # Scenario Q7: String split + arrays
│   └── q08_convert_date_formats.py    # Scenario Q8: Coalesce + to_date()
│
├── 04_aggregations_windows/           # SCENARIO Q9-Q15: Aggregations & ranking
│   ├── q09_running_total.py           # Scenario Q9: Window cumulative sum
│   ├── q10_top_n_per_group.py         # Scenario Q10: Dense rank + filter
│   ├── q11_mom_growth.py              # Scenario Q11: Lag + percent calculation
│   ├── q12_consecutive_months.py      # Scenario Q12: months_between + streaks
│   ├── q13_rolling_average.py         # Scenario Q13: rowsBetween + rangeBetween
│   ├── q14_percentile_buckets.py      # Scenario Q14: ntile + percent_rank
│   └── q15_second_highest_salary.py   # Scenario Q15: Dense rank filtering
│
├── 05_joins_set_ops/                  # SCENARIO Q16-Q20: Joins & set operations
│   ├── q16_anti_join.py               # Scenario Q16: Left anti + exists
│   ├── q17_union_distinct.py          # Scenario Q17: Union + dedup
│   ├── q18_fuzzy_join.py              # Scenario Q18: Levenshtein distance
│   ├── q19_scd_type2.py               # Scenario Q19: Change detection (SCD Type 2)
│   └── q20_skewed_join.py             # Scenario Q20: Salting + broadcast
│
├── 06_performance_optimization/       # SCENARIO Q21-Q25: Performance & tuning
│   ├── q21_diagnose_slow_jobs.py      # Scenario Q21: Explain plans + diagnostics
│   ├── q22_efficient_csv_read.py      # Scenario Q22: Large CSV (500GB+) read
│   ├── q23_partitioned_writes.py      # Scenario Q23: Output partitioning
│   ├── q24_predicate_pushdown.py      # Scenario Q24: Filter optimization
│   └── q25_large_scale_processing.py  # Scenario Q25: 1B+ row processing
│
└── 07_business_scenarios/             # SCENARIO Q26-Q30: Real-world use cases
    ├── q26_churn_detection.py         # Scenario Q26: Churn analysis
    ├── q27_session_window.py          # Scenario Q27: Session grouping
    ├── q28_anomaly_detection.py       # Scenario Q28: Anomaly detection
    ├── q29_data_reconciliation.py     # Scenario Q29: Data validation
    └── q30_daily_report_generation.py # Scenario Q30: Multi-level aggregation

### Configuration Files
```
├── README.md                          # This file (learning guide)
├── FILE_STRUCTURE_GUIDE.md            # File numbering & naming reference
├── COMPLETION_SUMMARY.md              # Project completion & statistics
├── requirements.txt                   # Dependencies: pyspark==3.5.0, pandas, numpy
└── .gitignore                         # Git ignore rules for Python + PySpark
```

---

## 🧪 Test Suite

**Comprehensive pytest coverage for all 36 solutions**

```bash
# Run all tests
pytest tests/ -v

# Run specific category
pytest tests/test_03_data_cleaning/ -v

# Run with coverage report
pytest tests/ --cov --cov-report=html
```

**Test Statistics:**
- **36 test files** with **275+ test cases**
- **100% coverage** of all solution files
- Organized by category matching solution folders
- Assertions validate output correctness

See [tests/README_TESTS.md](tests/README_TESTS.md) for detailed test documentation.

---

## 💻 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Java 8 or 11 (for Spark)
- 4GB RAM minimum

### Installation

#### Option 1: Using pip (Recommended)

```bash
# Create virtual environment
python -m venv spark-env

# Activate virtual environment
# On Windows:
spark-env\Scripts\activate
# On macOS/Linux:
source spark-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Using Conda

```bash
conda create -n pyspark-prep python=3.9
conda activate pyspark-prep
pip install -r requirements.txt
```

#### Option 3: Docker

```bash
# Pull Spark Python image
docker pull jupyter/all-spark-notebook:latest

# Run container
docker run -it -p 8888:8888 -v $(pwd):/home/jovyan/work jupyter/all-spark-notebook
```

### Verify Installation

```python
# Quick test
python
>>> from pyspark.sql import SparkSession
>>> spark = SparkSession.builder.appName("test").master("local[*]").getOrCreate()
>>> df = spark.createDataFrame([(1,"a"),(2,"b")], ["id","val"])
>>> df.show()
+---+---+
| id|val|
+---+---+
|  1|  a|
|  2|  b|
+---+---+
>>> spark.stop()
```

---

## 🏃 How to Run

### Run Individual Scripts

```bash
# Run a data cleaning example
python 03_data_cleaning/q01_remove_duplicates.py

# Run an aggregations example
python 04_aggregations_windows/q09_running_total.py

# Run a business scenario
python 07_business_scenarios/q26_churn_detection.py
```

### Run with Spark Submit

```bash
# Using spark-submit for distributed execution
spark-submit 03_data_cleaning/q01_remove_duplicates.py

# With custom config
spark-submit \
  --master local[4] \
  --driver-memory 2g \
  --executor-memory 2g \
  03_data_cleaning/q01_remove_duplicates.py
```

### Run Pytest Test Suite

```bash
# Run all unit tests
pytest tests/ -v

# Run tests for specific category
pytest tests/test_03_data_cleaning/ -v

# Run specific test file
pytest tests/test_03_data_cleaning/test_q01_remove_duplicates.py -v

# Run with coverage report
pytest tests/ --cov --cov-report=html
```

### Run Individual Solutions

```bash
# Create a test runner script
for file in */q*.py; do
    echo "Running: $file"
    python "$file"
    echo "---"
done
```

---

## 📚 Question Index

### 📖 PART 1: Theory Questions (30 Questions)

#### Section 1: Core Concepts (Q1-Q8)
1. What is Apache Spark and how does PySpark relate to it?
2. What is an RDD?
3. What is the difference between RDD, DataFrame, and Dataset?
4. What is lazy evaluation in Spark?
5. Explain the Spark Architecture.
6. What is a DAG in Spark?
7. What is the difference between `map()` and `flatMap()`?
8. What is the difference between Transformations and Actions?

#### Section 2: DataFrames & Spark SQL (Q9-Q16)
9. How do you create a SparkSession?
10. What is the difference between `select()` and `withColumn()`?
11. How do you handle null values in PySpark?
12. What is the difference between `cache()` and `persist()`?
13. Explain `groupBy()` vs `partitionBy()`
14. What are Window Functions? Give an example.
15. How do you perform joins in PySpark?
16. What is the difference between `orderBy()` and `sortWithinPartitions()`?

#### Section 3: Performance & Optimization (Q17-Q23)
17. What is a shuffle in Spark and why is it expensive?
18. What is broadcast join and when should you use it?
19. What is data skew and how do you handle it?
20. What is the difference between `repartition()` and `coalesce()`?
21. What is the Catalyst Optimizer?
22. How do you use `explain()` for query optimization?
23. What are Accumulators and Broadcast Variables?

#### Section 4: File Formats & I/O (Q24-Q27)
24. What file formats does PySpark support?
25. How do you read and write data in PySpark?
26. What is schema inference and why avoid it in production?
27. What is Delta Lake?

#### Section 5: Advanced Topics (Q28-Q30)
28. What is Structured Streaming in PySpark?
29. How does fault tolerance work in Spark?
30. What is the difference between Client Mode and Cluster Mode?

---

### 💡 PART 2: Scenario-Based Coding Questions (30 Questions)

#### Section 1: Data Cleaning & Transformation (Q1-Q8)
| # | Question | File | Concepts |
|---|----------|------|----------|
| Q1 | Remove duplicates & keep latest | [q01_remove_duplicates.py](03_data_cleaning/q01_remove_duplicates.py) | Window, row_number, rank |
| Q2 | Flatten nested JSON/struct | [q02_flatten_nested_json.py](03_data_cleaning/q02_flatten_nested_json.py) | StructType, select, recursion |
| Q3 | Pivot table transformation | [q03_pivot_table.py](03_data_cleaning/q03_pivot_table.py) | groupBy, pivot, aggregate |
| Q4 | Unpivot (melt) columns | [q04_unpivot_melt.py](03_data_cleaning/q04_unpivot_melt.py) | stack, union, values |
| Q5 | Forward fill nulls | [q05_forward_fill.py](03_data_cleaning/q05_forward_fill.py) | Window, last, ignorenulls |
| Q6 | Standardize phone numbers | [q06_standardize_phone_numbers.py](03_data_cleaning/q06_standardize_phone_numbers.py) | regexp_replace, substring |
| Q7 | Parse full name | [q07_parse_full_name.py](03_data_cleaning/q07_parse_full_name.py) | split, array indexing |
| Q8 | Convert multiple date formats | [q08_convert_date_formats.py](03_data_cleaning/q08_convert_date_formats.py) | coalesce, to_date |

#### Section 2: Aggregations & Window Functions (Q9-Q15)
| # | Question | File | Concepts |
|---|----------|------|----------|
| Q9 | Running total (cumsum) | [q09_running_total.py](04_aggregations_windows/q09_running_total.py) | Window, unboundedPreceding, sum |
| Q10 | Top N per category | [q10_top_n_per_group.py](04_aggregations_windows/q10_top_n_per_group.py) | dense_rank, filter, partitionBy |
| Q11 | Month-over-month growth % | [q11_mom_growth.py](04_aggregations_windows/q11_mom_growth.py) | lag, datediff, percent calc |
| Q12 | Consecutive months | [q12_consecutive_months.py](04_aggregations_windows/q12_consecutive_months.py) | months_between, gap detection |
| Q13 | 7-day rolling average | [q13_rolling_average.py](04_aggregations_windows/q13_rolling_average.py) | rowsBetween, rangeBetween, avg |
| Q14 | Percentile buckets/quartiles | [q14_percentile_buckets.py](04_aggregations_windows/q14_percentile_buckets.py) | ntile, percent_rank, cume_dist |
| Q15 | Second highest salary | [q15_second_highest_salary.py](04_aggregations_windows/q15_second_highest_salary.py) | dense_rank, rank, row_number |

#### Section 3: Joins & Set Operations (Q16-Q20)
| # | Question | File | Concepts |
|---|----------|------|----------|
| Q16 | Anti join (A not in B) | [q16_anti_join.py](05_joins_set_ops/q16_anti_join.py) | left_anti, exists, NOT IN |
| Q17 | Union distinct | [q17_union_distinct.py](05_joins_set_ops/q17_union_distinct.py) | union, dropDuplicates, unionByName |
| Q18 | Fuzzy join (name matching) | [q18_fuzzy_join.py](05_joins_set_ops/q18_fuzzy_join.py) | levenshtein, crossJoin, soundex |
| Q19 | SCD Type 2 (detect changes) | [q19_scd_type2.py](05_joins_set_ops/q19_scd_type2.py) | left_join, filter, history |
| Q20 | Skewed join optimization | [q20_skewed_join.py](05_joins_set_ops/q20_skewed_join.py) | salting, broadcast, partitioning |

#### Section 4: Performance & Optimization (Q21-Q25)
| # | Question | File | Concepts |
|---|----------|------|----------|
| Q21 | Diagnose slow jobs | 06_performance_optimization/q21_diagnose_slow_jobs.py | explain, partitions, UI |
| Q22 | Efficient large CSV read | 06_performance_optimization/q22_efficient_csv_read.py | schema, inferSchema, options |
| Q23 | Partitioned write strategy | 06_performance_optimization/q23_partitioned_writes.py | partitionBy, coalesce, output |
| Q24 | Predicate pushdown | 06_performance_optimization/q24_predicate_pushdown.py | filters, partition pruning |
| Q25 | Large scale processing | 06_performance_optimization/q25_large_scale_processing.py | filter, project, aggregate, join |

#### Section 5: Business Scenarios (Q26-Q30)
| # | Question | File | Concepts |
|---|----------|------|----------|
| Q26 | Churn detection | [q26_churn_detection.py](07_business_scenarios/q26_churn_detection.py) | max, datediff, case_when |
| Q27 | Session window | 07_business_scenarios/q27_session_window.py | unix_timestamp, gaps, session_id |
| Q28 | Anomaly detection | 07_business_scenarios/q28_anomaly_detection.py | stddev, z-score, abs |
| Q29 | Data reconciliation | 07_business_scenarios/q29_data_reconciliation.py | left_anti, except, mismatch |
| Q30 | Daily report generation | 07_business_scenarios/q30_daily_report_generation.py | multi-level agg, groupBy |

---

## 🎓 Key Concepts

### Transformations vs Actions

| Type | Examples | Returns |
|------|----------|---------|
| **Transformations** (Lazy) | `map`, `filter`, `select`, `join`, `groupBy` | New DataFrame/RDD |
| **Actions** (Eager) | `show`, `collect`, `count`, `write`, `cache` | Value or Side Effect |

### DataFrame Operations

```python
# Selection & Projection
df.select("col1", "col2")
df.select(df.col1, F.col("col2"))

# Filtering
df.filter(F.col("age") > 25)
df.where("salary > 100000")

# Grouping & Aggregation
df.groupBy("category").agg(F.sum("amount"))
df.groupBy("dept").count()

# Window Functions
from pyspark.sql.window import Window
window = Window.partitionBy("dept").orderBy("salary")
df.withColumn("rank", F.row_number().over(window))

# Joins
df1.join(df2, on="id", how="inner")
df1.join(F.broadcast(df2), on="id")  # Broadcast join

# Null Handling
df.dropna()
df.fillna({"col1": 0, "col2": "Unknown"})
df.filter(F.col("col").isNotNull())
```

### Common Window Functions

```python
# Ranking
F.row_number()    # 1, 2, 3, 4, 5
F.rank()          # 1, 2, 2, 4, 5 (gaps on ties)
F.dense_rank()    # 1, 2, 2, 3, 4 (no gaps)

# Lead/Lag
F.lag("col", 1)   # Previous row
F.lead("col", 1)  # Next row

# Aggregation
F.sum("col").over(window)   # Running total
F.avg("col").over(window)   # Running average
F.first("col").over(window) # First value in window
F.last("col").over(window)  # Last value in window

# Distribution
F.ntile(4)        # Quartiles
F.percent_rank()  # 0.0 to 1.0
F.cume_dist()     # Cumulative distribution
```

### Performance Tips

```python
# 1. Predicate Pushdown (filter early)
df.filter(F.col("year") == 2024)  # Before read

# 2. Column Pruning (select only needed)
df.select("id", "name", "amount")

# 3. Broadcasting (for small tables)
df_large.join(F.broadcast(df_small), on="key")

# 4. Caching (for reused DataFrames)
df.cache()
df.count()  # Triggers cache

# 5. Partitioning (for write)
df.write.partitionBy("year", "month").parquet("path")

# 6. Repartition (for processing)
df.repartition(200)  # Increases partitions (shuffle)
df.coalesce(10)      # Decreases partitions (no shuffle)

# 7. SQL Optimizer
spark.sql("SELECT ... WHERE ...")  # Uses Catalyst optimizer
```

---

## 📊 Spark Configuration Tips

```python
# Session config
spark = SparkSession.builder \
    .appName("MyApp") \
    .master("local[4]") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Common settings
spark.conf.set("spark.sql.shuffle.partitions", "200")  # Tune for data size
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10mb")  # Auto broadcast
spark.conf.set("spark.sql.adaptive.enabled", "true")  # Adaptive optimization (Spark 3.0+)
spark.conf.set("spark.memory.fraction", "0.8")  # Memory allocation
spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")  # Kryo
```

---

## 📝 Example Usage

### Running a Simple Example

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize
spark = SparkSession.builder.appName("Example").master("local[*]").getOrCreate()

# Create sample data
data = [(1, "Alice", 30), (2, "Bob", 25), (3, "Carol", 35)]
df = spark.createDataFrame(data, ["id", "name", "age"])

# Show data
df.show()

# Perform operations
result = df.filter(F.col("age") > 25).select("name", "age")
result.show()

# Stop SparkSession
spark.stop()
```

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Submit a Pull Request

---

## 📚 Additional Resources

- [PySpark Official Documentation](https://spark.apache.org/docs/latest/api/python/)
- [Databricks Academy](https://academy.databricks.com/)
- [Spark by Examples](https://sparkbyexamples.com/)
- [Medium - Towards Data Science](https://towardsdatascience.com/)

---

## ⚠️ Important Notes

1. **Local Mode**: These examples run in local mode (`master="local[*]"`). For cluster deployment, change to `master="yarn"` or `master="k8s://..."`.

2. **Data Size**: For production with large datasets (100GB+), ensure sufficient cluster resources and tune configurations accordingly.

3. **Java Requirement**: PySpark requires Java 8 or 11. Ensure `JAVA_HOME` is set correctly.

4. **Memory**: Adjust `--driver-memory` and `--executor-memory` based on your machine's resources.

---

## 📞 Support

- **Questions**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions
- **Feedback**: Submit feedback via pull requests

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🎯 Quick Tips for Interview Success

1. **Understand DAGs**: Always think about Spark's execution plan
2. **Know the difference**: RDD vs DataFrame, map vs flatMap, cache vs persist
3. **Window Functions**: Master these - they appear in many questions
4. **Joins**: Understand different types and when to broadcast
5. **Optimization**: Always think about data size, shuffles, and stage boundaries
6. **Explain Plans**: Use `df.explain()` to understand how Spark optimizes your code
7. **Testing**: Run examples locally before production
8. **Performance**: Monitor for shuffes, stages, and task durations

---

**Last Updated**: March 2024  
**Version**: 1.0  
**Author**: PySpark Interview Prep Community

---

### ⭐ If this helps you ace your interview, please give it a star!

# pyspark-interview-masterclass
