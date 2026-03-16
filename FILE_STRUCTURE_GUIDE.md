# 📋 File Structure & Numbering Guide

## Overview

This project contains **36 total files**:
- **6 BONUS files** (not in original 30): Core Concepts + DataFrames  
- **30 Original Scenario Files**: Data Cleaning through Business Scenarios

## Numbering Scheme

### Files 01-06: BONUS (Extra Learning Resources)

```
01_core_concepts/          (BONUS - Extra Files)
├── q01_rdd_basics.py                    → Topic: RDD fundamentals
├── q02_transformations_actions.py       → Topic: Lazy vs Eager execution  
└── q03_sparksession.py                  → Topic: SparkSession configuration

02_dataframes_sql/         (BONUS - Extra Files)
├── q04_select_withcolumn.py             → Topic: Select & withColumn operations
├── q05_null_handling.py                 → Topic: Null value handling
└── q06_window_functions.py              → Topic: Window function deep dive
```

These 6 files are **BONUS** educational files that provide foundational knowledge before the main 30 scenario questions.

---

### Scenario Files 01-30: Original Interview Questions

```
03_data_cleaning/          (Scenario Q1-Q8)
├── q01_remove_duplicates.py             → Scenario Q1: Deduplication
├── q02_flatten_nested_json.py           → Scenario Q2: JSON flattening
├── q03_pivot_table.py                   → Scenario Q3: Pivot operations
├── q04_unpivot_melt.py                  → Scenario Q4: Unpivoting
├── q05_forward_fill.py                  → Scenario Q5: Forward fill nulls
├── q06_standardize_phone_numbers.py     → Scenario Q6: Data normalization
├── q07_parse_full_name.py               → Scenario Q7: String parsing
└── q08_convert_date_formats.py          → Scenario Q8: Date conversion

04_aggregations_windows/   (Scenario Q9-Q15)
├── q09_running_total.py                 → Scenario Q9: Running totals
├── q10_top_n_per_group.py               → Scenario Q10: Top N ranking
├── q11_mom_growth.py                    → Scenario Q11: MoM growth %
├── q12_consecutive_months.py            → Scenario Q12: Streaks/sequences
├── q13_rolling_average.py               → Scenario Q13: Rolling windows
├── q14_percentile_buckets.py            → Scenario Q14: Percentile ranking
└── q15_second_highest_salary.py         → Scenario Q15: Dense ranking

05_joins_set_ops/          (Scenario Q16-Q20)
├── q16_anti_join.py                     → Scenario Q16: Anti joins
├── q17_union_distinct.py                → Scenario Q17: Set operations
├── q18_fuzzy_join.py                    → Scenario Q18: Fuzzy matching
├── q19_scd_type2.py                     → Scenario Q19: Change detection
└── q20_skewed_join.py                   → Scenario Q20: Skew optimization

06_performance_optimization/ (Scenario Q21-Q25)
├── q21_diagnose_slow_jobs.py            → Scenario Q21: Query diagnostics
├── q22_efficient_csv_read.py            → Scenario Q22: Large file handling
├── q23_partitioned_writes.py            → Scenario Q23: Output partitioning
├── q24_predicate_pushdown.py            → Scenario Q24: Filter optimization
└── q25_large_scale_processing.py        → Scenario Q25: 1B+ row processing

07_business_scenarios/     (Scenario Q26-Q30)
├── q26_churn_detection.py               → Scenario Q26: Churn analysis
├── q27_session_window.py                → Scenario Q27: Session grouping
├── q28_anomaly_detection.py             → Scenario Q28: Anomaly detection
├── q29_data_reconciliation.py           → Scenario Q29: Data validation
└── q30_daily_report_generation.py       → Scenario Q30: Report generation
```

---

## File Naming Convention

**Format**: `q{NUMBER}_{description}.py`

- **q01-q08**: Scenario questions 1-8 (Data Cleaning folder)
- **q09-q15**: Scenario questions 9-15 (Aggregations folder)
- **q16-q20**: Scenario questions 16-20 (Joins folder)
- **q21-q25**: Scenario questions 21-25 (Performance folder)
- **q26-q30**: Scenario questions 26-30 (Business folder)

Bonus files in separate folders:
- **q01-q03**: Core concepts (01_core_concepts folder)
- **q04-q06**: DataFrames & SQL (02_dataframes_sql folder)

---

## Running the Files

### Run a Specific Scenario (e.g., Q1 - Remove Duplicates)
```bash
python 03_data_cleaning/q01_remove_duplicates.py
```

### Run a Specific Bonus File (e.g., RDD Basics)
```bash
python 01_core_concepts/q01_rdd_basics.py
```

### Run All Files in a Category
```bash
# Data Cleaning scenarios
for file in 03_data_cleaning/*.py; do python "$file"; done

# All aggregation scenarios  
for file in 04_aggregations_windows/*.py; do python "$file"; done
```

---

## File Count by Category

| Category | Files | Type | Scenarios |
|----------|-------|------|-----------|
| 01_core_concepts | 3 | BONUS | — |
| 02_dataframes_sql | 3 | BONUS | — |
| 03_data_cleaning | 8 | Original | Q1-Q8 |
| 04_aggregations_windows | 7 | Original | Q9-Q15 |
| 05_joins_set_ops | 5 | Original | Q16-Q20 |
| 06_performance_optimization | 5 | Original | Q21-Q25 |
| 07_business_scenarios | 5 | Original | Q26-Q30 |
| **TOTAL** | **36** | **6 BONUS + 30 Original** | **Q1-Q30** |

---

## Quality Checklist ✅

Each file contains:
- ✅ Complete working PySpark code
- ✅ Sample input data
- ✅ Multiple solution approaches
- ✅ Detailed inline comments
- ✅ Expected output in comments
- ✅ Best practices & optimization tips
- ✅ Real-world scenario context

---

## How to Use This Guide

1. **For interviews**: Study Q1-Q30 scenarios in order (folders 03-07)
2. **For foundations**: Start with bonus files (folders 01-02)
3. **For reference**: Use FILE_STRUCTURE_GUIDE.md to locate topics
4. **For verification**: All files follow naming convention: `q{num}_{topic}.py`
