# Probabilistic Databases with Data Imputation

This project implements querying over **uncertain and incomplete datasets**, combining **probabilistic databases** and **data imputation**. The goal is to handle missing values, assign or update tuple probabilities, and execute supported queries using either **independent** or **dependent** probabilistic models.

---

## 📦 Project Structure

probabilistic_db_project/
│
├── data/
│   └── input_dataset.csv              # Your input file (can be probabilistic or not)
│
├── preprocessing/
│   ├── imputer.py                     # Handles missing value imputation (KNN, RF)
│   ├── assign_probability.py          # Assigns or updates tuple probabilities
│   ├── missingness_detector.py        # Classifies MCAR, MAR, MNAR patterns
│   └── utils.py                       # Shared helpers (e.g., data loading, logging)
│
├── query_engine/
│   ├── query_parser.py                # Parses and validates conjunctive/UCQ queries
│   ├── evaluator.py                   # Executes queries under independent/dependent model
│   ├── possible_worlds.py             # Generates and evaluates possible worlds
│   └── safety_checker.py              # Implements Find-Plan(q) algorithm
│
├── models/
│   └── dependent_case_bayesian_network.py  # Bayesian Network model for dependent mode
│   ├── independent_case.py                 # Executes queries under independent model
│
├── config/
│   └── settings.yaml                  # Config file to control modes, input paths, model types
│
├── examples/
│   └── input_queries.py               # Step-by-step input queries with different operations
│
├── output/
│   └── results.json                   # Final query output with probabilities
│   └── result_analysis.py             # Measures the total runtime and output quality
│
├── run_pipeline.py                    # Main entry point
└── README.md                          # Project documentation

## 🔄 Pipeline Overview

### ✅ Input Types Supported:
- **Probabilistic datasets** (with a probability column)
- **Non-probabilistic datasets** (with only missing values)

### ✅ Missing Data Types Handled:
- MCAR (random)
- MAR (depends on other attribute)
- MNAR (depends on missing value itself)

### ✅ Preprocessing:
- **If dataset is probabilistic**:
  - Impute missing values
  - Update tuple probability (Bayesian update)
- **If dataset is non-probabilistic**:
  - Impute missing values
  - Assign probabilities (uniform or Bayesian)

---

### ✅ Query Support:

#### Supported Operations:
- Selection (σ)
- Projection (π)
- Join (⨝)
- Union (∪)
- Disjoint Projection (πᴰ)

#### Supported Query Types:
- **Conjunctive Queries** (AND only)
- **UCQ (Union of Conjunctive Queries)**

#### Output:
- Results are probabilistic: each row has a `probability` field.

---

### ✅ Probabilistic Model Modes

#### 1. Independent Model:
- Multiply tuple probabilities.
- Fast and efficient.

#### 2. Dependent Model:
- Use Bayesian Network to compute conditional probabilities.
- Handles real-world dependencies.

---

### ✅ Query Safety Analysis (Find-Plan(q)):
- Implemented in `safety_checker.py`
- Determines if query is **safe (efficient)** or **unsafe (#P-complete)**.
- If unsafe, recommend dependent model.

---

## 🚀 How to Run

### 1. Setup
pip install -r requirements.txt

2. Configure
Edit the file: config/settings.yaml

3. Run Full Pipeline
python run_pipeline.py --config config/settings.yaml

✅ Example Output

[
  {
    "Employee_ID": 1,
    "Name": "Alice",
    "Department": "HR",
    "Salary": 6000,
    "probability": 0.87
  }
]


🙌 Credits
Project by Haneen Najjar
Course: Advanced Topics in Database Management
Tel Aviv University – Spring 2025