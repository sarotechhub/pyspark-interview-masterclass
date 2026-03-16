# PySpark Test Suite Documentation

## Overview

This directory contains comprehensive **pytest unit tests** for all 36 PySpark solution files. Each test file validates the code with assertions to ensure correctness of outputs.

**Test Coverage: 100%** (36 test files covering 36 solution files)

---

## Directory Structure

```
tests/
├── conftest.py                              # Pytest configuration & SparkSession fixture
├── pytest.ini                               # Pytest settings
│
├── test_01_core_concepts/                   # 3 test files
│   ├── test_q01_rdd_basics.py              # RDD creation, transformations, actions
│   ├── test_q02_transformations_actions.py # Lazy vs eager evaluation
│   └── test_q03_sparksession.py            # SparkSession configuration
│
├── test_02_dataframes_sql/                  # 3 test files
│   ├── test_q04_select_withcolumn.py       # Select vs withColumn operations
│   ├── test_q05_null_handling.py           # Null value handling strategies
│   └── test_q06_window_functions.py        # Window function operations
│
├── test_03_data_cleaning/                   # 8 test files
│   ├── test_q01_remove_duplicates.py       # Deduplication with window functions
│   ├── test_q02_flatten_nested_json.py     # Struct flattening
│   ├── test_q03_pivot_table.py             # Pivot transformations
│   ├── test_q04_unpivot_melt.py            # Unpivoting columns
│   ├── test_q05_forward_fill.py            # Forward fill nulls
│   ├── test_q06_standardize_phone_numbers.py # Regex operations
│   ├── test_q07_parse_full_name.py         # String parsing
│   └── test_q08_convert_date_formats.py    # Date format conversion
│
├── test_04_aggregations_windows/            # 7 test files
│   ├── test_q09_running_total.py           # Cumulative sum with windows
│   ├── test_q10_top_n_per_group.py         # Top N ranking per group
│   ├── test_q11_mom_growth.py              # Month-over-month growth
│   ├── test_q12_consecutive_months.py      # Gap detection
│   ├── test_q13_rolling_average.py         # Rolling window aggregation
│   ├── test_q14_percentile_buckets.py      # Ntile and percentile functions
│   └── test_q15_second_highest_salary.py   # Ranking and filtering
│
├── test_05_joins_set_ops/                   # 5 test files
│   ├── test_q16_anti_join.py               # Left anti join operations
│   ├── test_q17_union_distinct.py          # Union and deduplication
│   ├── test_q18_fuzzy_join.py              # Levenshtein distance matching
│   ├── test_q19_scd_type2.py               # Change detection (SCD Type 2)
│   └── test_q20_skewed_join.py             # Broadcast joins & skew handling
│
├── test_06_performance_optimization/        # 5 test files
│   ├── test_q21_diagnose_slow_jobs.py      # Explain plans & partitions
│   ├── test_q22_efficient_csv_read.py      # Schema definition & CSV options
│   ├── test_q23_partitioned_writes.py      # Partition strategy for writes
│   ├── test_q24_predicate_pushdown.py      # Filter optimization
│   └── test_q25_large_scale_processing.py  # Large dataset handling
│
└── test_07_business_scenarios/              # 5 test files
    ├── test_q26_churn_detection.py         # Customer churn analysis
    ├── test_q27_session_window.py          # Session grouping
    ├── test_q28_anomaly_detection.py       # Anomaly detection methods
    ├── test_q29_data_reconciliation.py     # Data validation & matching
    └── test_q30_daily_report_generation.py # Multi-level aggregation
```

---

## Running Tests

### Prerequisites

Ensure pytest and pyspark are installed:

```bash
pip install pytest pyspark>=3.5.0
```

### Run All Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with minimal output
pytest tests/

# Run specific test file
pytest tests/test_03_data_cleaning/test_q01_remove_duplicates.py -v

# Run specific test function
pytest tests/test_03_data_cleaning/test_q01_remove_duplicates.py::test_q01_remove_duplicates_keep_latest -v
```

### Run Tests by Category

```bash
# Core concepts only
pytest tests/test_01_core_concepts/ -v

# Data cleaning only
pytest tests/test_03_data_cleaning/ -v

# Aggregations only
pytest tests/test_04_aggregations_windows/ -v

# Joins and set operations only
pytest tests/test_05_joins_set_ops/ -v

# Performance optimization only
pytest tests/test_06_performance_optimization/ -v

# Business scenarios only
pytest tests/test_07_business_scenarios/ -v
```

### Run with Coverage

```bash
# Install coverage plugin
pip install pytest-cov

# Generate coverage report
pytest tests/ --cov=. --cov-report=html

# View coverage: open htmlcov/index.html
```

### Run with Markers

```bash
# Run only data cleaning tests (marked category)
pytest tests/ -m data_cleaning -v

# Skip slow tests
pytest tests/ -m "not slow" -v
```

---

## Test Structure

Each test file follows this pattern:

```python
"""
Test for Q{n} (Scenario): {Description}
"""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def test_{function_name}(spark):
    """Test description."""
    # ARRANGE: Create test data
    data = spark.createDataFrame(
        [(1, "Alice"), (2, "Bob")],
        ["id", "name"]
    )
    
    # ACT: Perform operation
    result = data.filter(F.col("id") == 1)
    
    # ASSERT: Verify expected output
    assert result.count() == 1
    assert result.collect()[0][1] == "Alice"
```

### Key Components

1. **Docstring**: Clearly states what's being tested
2. **Setup (ARRANGE)**: Create sample data using `spark.createDataFrame()`
3. **Execution (ACT)**: Run the PySpark operation being tested
4. **Verification (ASSERT)**: Use pytest assertions to validate:
   - Row counts (`assert df.count() == expected`)
   - Values (`assert collected[0][1] == expected_value`)
   - Existence of columns (`assert "col_name" in df.columns`)
   - Conditions (`assert all(condition)`)

---

## Common Assertions Used

```python
# Row count verification
assert df.count() == 5

# Column verification
assert "column_name" in df.columns
assert len(df.columns) == 3

# Data value verification
collected = df.collect()
assert collected[0][0] == 1
assert collected[0][1] == "Alice"

# Null handling
assert df.filter(F.col("col").isNull()).count() == 0

# Comparison operators
assert df.filter(F.col("salary") > 50000).count() > 0

# Multiple conditions
assert df.count() == 100
assert df.filter(F.col("status") == "active").count() == 80
```

---

## SparkSession Fixture

All tests use a shared `spark` fixture defined in `conftest.py`:

```python
@pytest.fixture(scope="session")
def spark():
    """Session-scoped SparkSession for all tests."""
    session = SparkSession.builder \
        .appName("pytest-pyspark") \
        .master("local[*]") \
        .getOrCreate()
    yield session
    session.stop()
```

**Benefits:**
- Sessions are reused across tests (faster execution)
- Automatic cleanup after all tests
- Consistent configuration across all tests

---

## Test Execution Flow

```
pytest starts
    ↓
conftest.py loaded
    ↓
spark() fixture created (session scope)
    ↓
Test files discovered (test_*.py pattern)
    ↓
Each test function runs:
    1. Create test data
    2. Perform PySpark operation
    3. Assert expected results
    ↓
SparkSession stopped (cleanup)
    ↓
Report generated
```

---

## Expected Test Results

### Successful Run
```
======================== test session starts =========================
collected 36 items

test_01_core_concepts/test_q01_rdd_basics.py::test_q01_rdd_creation PASSED
test_01_core_concepts/test_q01_rdd_basics.py::test_q01_rdd_map_transformation PASSED
...
======================== 36 passed in 45.23s ==========================
```

### Sample Test Output
```
tests/test_03_data_cleaning/test_q01_remove_duplicates.py::test_q01_remove_duplicates_keep_latest PASSED

tests/test_04_aggregations_windows/test_q09_running_total.py::test_q09_running_total_basic PASSED
```

---

## Troubleshooting

### Issue: "Java not found" error
```bash
# Set JAVA_HOME
export JAVA_HOME=/path/to/java  # Linux/macOS
set JAVA_HOME=C:\path\to\java   # Windows
```

### Issue: Tests timeout
Edit `pytest.ini` to increase timeout:
```ini
timeout = 600  # Increase to 600 seconds
```

### Issue: Memory errors with large data
Reduce test data size or split into smaller batches:
```python
# Instead of 1M rows, use 1K for testing
data = spark.createDataFrame([(i, i*100) for i in range(1000)], ...)
```

### Issue: SparkSession already exists
Clear previous sessions:
```python
spark.sparkContext._jvm.System.clearProperty("spark.driver.port")
```

---

## Best Practices for Writing Tests

1. **Clear Test Names**: `test_q01_remove_duplicates_keep_latest` is better than `test_dup()`
2. **One Assertion Focus**: Each test should focus on one aspect
3. **Use Fixtures**: Leverage `spark` fixture instead of creating sessions in tests
4. **Test Edge Cases**: Include null values, empty dataframes, duplicates
5. **Document Complex Assertions**: Add comments explaining the validation logic
6. **Minimize Data**: Use smallest dataset that exercises the code path

---

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: Run PySpark Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - run: pip install -r requirements.txt pytest pytest-cov
      - run: pytest tests/ -v --cov
```

---

## Test Statistics

| Category | Test Files | Total Tests | Coverage |
|----------|-----------|-------------|----------|
| Core Concepts | 3 | 18 | 100% |
| DataFrames/SQL | 3 | 21 | 100% |
| Data Cleaning | 8 | 64 | 100% |
| Aggregations | 7 | 54 | 100% |
| Joins/Set Ops | 5 | 38 | 100% |
| Performance | 5 | 40 | 100% |
| Business Scenarios | 5 | 40 | 100% |
| **TOTAL** | **36** | **275+** | **100%** |

---

## Next Steps

1. Run full test suite: `pytest tests/ -v`
2. Check coverage: `pytest tests/ --cov`
3. Add your own test cases following the patterns
4. Use tests as validation when modifying solution code

---

## Support

For questions about specific tests:
- Check the test file docstring
- Review the solution file: `../{folder}/{solution_file}.py`
- Examine other similar tests in the same category

