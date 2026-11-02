# Mammography Knowledge Discovery

### Machine Learning and Knowledge Discovery Based on CBIS-DDSM Clinical Descriptors

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)](https://jupyter.org/)

## 1. Project Overview

**Mammography Knowledge Discovery** is a machine learning and data mining project based on clinical descriptors from the CBIS-DDSM (Curated Breast Imaging Subset of DDSM) dataset.

The project explores two types of mammographic examinations:

- **Mass:** Clinical descriptors related to mammographic masses.
- **Calcification:** Clinical descriptors related to mammographic calcifications.

The objective is to extract useful patterns from clinical data, discover relationships between features, identify groups of similar cases, and develop machine learning models for pathology classification.

The project also includes semantic knowledge graphs built from selected association rules.

The final objective is to make these results available through a web application, allowing users to explore clinical descriptors, discover patterns, visualize relationships, and interact with the trained models.

> **Important:** This project is intended for educational and experimental purposes. It is not a validated medical diagnostic system and must not be used to make clinical decisions.

---

## 2. Project Objectives

The project is organized around the following objectives:

- Understand and explore the CBIS-DDSM clinical descriptor datasets.
- Clean and prepare clinical data for machine learning.
- Discover relationships between categorical features using association rule mining.
- Identify groups of similar cases using K-Means clustering.
- Train classification models to predict pathology categories.
- Evaluate models using internal and official test datasets.
- Build semantic knowledge graphs from selected association rules.
- Prepare reusable models and processed artifacts for integration into a web application.

---

## 3. Dataset

### Dataset Source

The project uses the CBIS-DDSM dataset:

**Kaggle:** [CBIS-DDSM Breast Cancer Image Dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)

This project focuses on the clinical descriptor CSV files. It does not perform direct mammography image classification.

### Dataset Files

| File | Description |
|---|---|
| `mass_case_description_train_set.csv` | Mass training dataset |
| `mass_case_description_test_set.csv` | Official Mass test dataset |
| `calc_case_description_train_set.csv` | Calcification training dataset |
| `calc_case_description_test_set.csv` | Official Calcification test dataset |
| `dicom_info.csv` | DICOM metadata |
| `meta.csv` | Additional metadata |

### Dataset Dimensions

| Dataset | Training Samples | Official Test Samples | Original Columns |
|---|---:|---:|---:|
| Mass | 1,318 | 378 | 14 |
| Calcification | 1,546 | 326 | 14 |

### Target Variable

The target variable is `pathology`, which contains three categories:

| Category | Encoded Value |
|---|---:|
| `BENIGN` | 0 |
| `BENIGN_WITHOUT_CALLBACK` | 1 |
| `MALIGNANT` | 2 |

The original category names should be preserved when displaying results in the application.

### Selected Features

**Mass features:**

- `breast_density`
- `left or right breast`
- `image view`
- `mass shape`
- `mass margins`
- `assessment`
- `subtlety`

**Calcification features:**

- `breast_density`
- `left or right breast`
- `image view`
- `calc type`
- `calc distribution`
- `assessment`
- `subtlety`

The `patient_id` is retained separately for patient-level splitting and integrity checks. It is not used as a model input feature.

---

## 4. Project Structure

```text
Mammography-Knowledge-Discovery/
│
├── data/
│   ├── mass_case_description_train_set.csv
│   ├── mass_case_description_test_set.csv
│   ├── calc_case_description_train_set.csv
│   ├── calc_case_description_test_set.csv
│   ├── dicom_info.csv
│   ├── meta.csv
│   │
│   └── processed/
│       ├── mass_X_train.npz
│       ├── mass_X_test.npz
│       ├── calc_X_train.npz
│       ├── calc_X_test.npz
│       ├── mass_y_train.npy
│       ├── mass_y_test.npy
│       ├── calc_y_train.npy
│       ├── calc_y_test.npy
│       ├── mass_patient_train.npy
│       ├── mass_patient_test.npy
│       ├── calc_patient_train.npy
│       ├── calc_patient_test.npy
│       ├── mass_train_idx.npy
│       ├── calc_train_idx.npy
│       ├── mass_clusters.npy
│       ├── calc_clusters.npy
│       ├── mass_association_rules.csv
│       ├── calc_association_rules.csv
│       ├── mass_selected_rules.csv
│       ├── calc_selected_rules.csv
│       ├── mass_knowledge_graph.graphml
│       └── calc_knowledge_graph.graphml
│
├── models/
│   ├── mass_preprocessor.joblib
│   ├── calc_preprocessor.joblib
│   ├── mass_logistic_regression.joblib
│   ├── mass_random_forest.joblib
│   ├── calc_logistic_regression.joblib
│   └── calc_random_forest.joblib
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_association_rules.ipynb
│   ├── 04_clustering.ipynb
│   ├── 05_classification.ipynb
│   ├── 06_semantic_graph.ipynb
│   ├── 07_overall_evaluation.ipynb
│   └── 08_official_test_evaluation.ipynb
│
├── main.py
├── README.md
└── .gitignore
```

*This structure describes the main project files and generated artifacts. Additional metadata and files may also be present.*

---

## 5. Methodology and Workflow

The project follows a Knowledge Discovery in Databases (KDD)-inspired workflow.

### 01 — Data Understanding

**Notebook:** `01_data_understanding.ipynb`

Initial exploration of the Mass and Calcification datasets.

Tasks:

- Inspect dataset dimensions and data types.
- Analyze missing values.
- Explore pathology distributions.
- Check duplicated records.
- Examine patient IDs.
- Analyze feature distributions.
- Study correlations between numerical features.

The initial exploration revealed missing values in several categorical descriptors and repeated patient IDs, making patient-level splitting important.

### 02 — Data Preparation

**Notebook:** `02_data_preparation.ipynb`

Preparation of the datasets for machine learning.

Tasks:

- Handle missing categorical values using `UNKNOWN`.
- Standardize inconsistent category names.
- Select relevant features.
- Encode categorical variables using `OneHotEncoder`.
- Fit preprocessing pipelines only on training data.
- Encode pathology categories.
- Split the data using `StratifiedGroupKFold`.

Patient IDs are used as groups during splitting to prevent the same patient from appearing in both the internal training and test subsets.

The fitted preprocessors are saved using Joblib:

- `mass_preprocessor.joblib`
- `calc_preprocessor.joblib`

### 03 — Association Rule Mining

**Notebook:** `03_association_rules.ipynb`

Association rule mining is used to discover relationships between categorical clinical descriptors.

Main steps:

1. Convert categorical features into transactions.
2. Extract frequent itemsets.
3. Generate association rules.
4. Filter rules using support, confidence and lift.

Filtering criteria:

| Metric | Threshold |
|---|---:|
| Support | ≥ 0.05 |
| Confidence | ≥ 0.70 |
| Lift | > 1.5 |

Generated files:

- `mass_association_rules.csv`
- `calc_association_rules.csv`

**Methodological note:** The current association rule notebook uses the original CSV data. Missing values and inconsistent categories may therefore affect the extracted rules. These results should be interpreted with this limitation in mind.

The `pathology` target is excluded from the association mining feature lists.

### 04 — Clustering

**Notebook:** `04_clustering.ipynb`

K-Means clustering is applied independently to Mass and Calcification training features.

Main steps:

- Use the Elbow Method to examine the number of clusters.
- Apply K-Means with K = 4.
- Analyze cluster sizes.
- Compare cluster composition with pathology categories after clustering.

Pathology is not used to fit the clustering algorithm. It is only used afterward to interpret the clusters.

Generated files:

- `mass_clusters.npy`
- `calc_clusters.npy`

### 05 — Classification

**Notebook:** `05_classification.ipynb`

Four classification models are trained:

| Dataset | Model |
|---|---|
| Mass | Logistic Regression |
| Mass | Random Forest |
| Calcification | Logistic Regression |
| Calcification | Random Forest |

Random Forest uses class weighting to help address class imbalance.

The trained models are saved in the `models/` directory.

### 06 — Semantic Knowledge Graph

**Notebook:** `06_semantic_graph.ipynb`

Selected association rules are transformed into directed semantic knowledge graphs.

The graphs represent relationships between clinical feature values and association rules.

Generated files:

- `mass_knowledge_graph.graphml`
- `calc_knowledge_graph.graphml`
- `mass_selected_rules.csv`
- `calc_selected_rules.csv`

Graph statistics:

| Dataset | Nodes | Edges | Feature Nodes | Rule Nodes |
|---|---:|---:|---:|---:|
| Mass | 21 | 36 | 11 | 10 |
| Calcification | 25 | 36 | 14 | 11 |

The GraphML files can be loaded using compatible graph-processing and visualization libraries.

### 07 — Overall Evaluation

**Notebook:** `07_overall_evaluation.ipynb`

This notebook consolidates the main experimental results:

- Classification performance.
- Clustering distributions.
- Pathology composition of clusters.
- Association rule statistics.
- Knowledge graph statistics.
- Comparison between Mass and Calcification.

### 08 — Official Test Evaluation

**Notebook:** `08_official_test_evaluation.ipynb`

The four trained models are evaluated using the official test CSV files provided by CBIS-DDSM.

Tasks:

1. Load official test datasets.
2. Clean and prepare the test features.
3. Load the previously fitted preprocessors.
4. Transform the official test data.
5. Load the trained classification models.
6. Calculate evaluation metrics.
7. Generate confusion matrices.
8. Analyze class-level precision, recall and F1-score.
9. Verify patient-level separation between the original training and official test datasets.

**Patient overlap verification:**

| Dataset | Shared Patients |
|---|---:|
| Mass | 0 |
| Calcification | 0 |

No patient overlap was detected between the original training and official test files.

---

## 6. Classification Results

### Official Test Evaluation

The following results were obtained using the official test datasets.

| Dataset | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---:|---:|---:|---:|
| Mass | Logistic Regression | 0.725 | 0.781 | 0.671 | 0.699 |
| Mass | Random Forest | 0.741 | 0.699 | 0.692 | 0.694 |
| Calcification | Logistic Regression | 0.702 | 0.757 | 0.745 | 0.742 |
| Calcification | Random Forest | 0.733 | 0.773 | 0.769 | 0.770 |

### Malignant Class Performance

| Dataset | Model | Precision | Recall | F1-score |
|---|---|---:|---:|---:|
| Mass | Logistic Regression | 0.64 | 0.85 | 0.73 |
| Mass | Random Forest | 0.72 | 0.80 | 0.76 |
| Calcification | Logistic Regression | 0.68 | 0.49 | 0.57 |
| Calcification | Random Forest | 0.70 | 0.64 | 0.67 |

### Observations

- Mass Logistic Regression achieves the highest malignant recall (85%).
- Mass Random Forest achieves the highest Mass accuracy (74.1%).
- Calcification Random Forest achieves the highest overall Calcification macro F1-score (0.770).
- Calcification Logistic Regression has a lower malignant recall (49%).

There is no single model that performs best across every dataset and evaluation metric.

These results are experimental and specific to the selected CBIS-DDSM descriptors and evaluation setup.

---

## 7. Models and Generated Artifacts

### Preprocessing Objects

| File | Description |
|---|---|
| `mass_preprocessor.joblib` | Mass feature preprocessing pipeline |
| `calc_preprocessor.joblib` | Calcification feature preprocessing pipeline |

### Classification Models

| File | Dataset | Algorithm |
|---|---|---|
| `mass_logistic_regression.joblib` | Mass | Logistic Regression |
| `mass_random_forest.joblib` | Mass | Random Forest |
| `calc_logistic_regression.joblib` | Calcification | Logistic Regression |
| `calc_random_forest.joblib` | Calcification | Random Forest |

### Clustering Artifacts

| File | Description |
|---|---|
| `mass_clusters.npy` | Mass training cluster assignments |
| `calc_clusters.npy` | Calcification training cluster assignments |

### Knowledge Graphs

| File | Description |
|---|---|
| `mass_knowledge_graph.graphml` | Mass semantic knowledge graph |
| `calc_knowledge_graph.graphml` | Calcification semantic knowledge graph |

### Association Rules

The `data/processed/` directory contains:

- Complete generated association rule exports.
- Selected association rules used for knowledge graph construction.

---

## 8. Web Application Integration Guide

This section is intended for developers integrating the trained models and generated artifacts into the web application.

### 8.1. Recommended Architecture

The trained models are Python scikit-learn objects saved as `.joblib` files.

They cannot be loaded and executed directly by a standard browser application.

A recommended architecture is:

```text
          HTML / CSS / JavaScript
                    |
                    | HTTP Requests
                    v
              Python Backend
                  FastAPI
                    |
          ---------------------
          |                   |
          v                   v
     Preprocessor         ML Model
          |                   |
          ---------------------
                    |
                    v
             Prediction Result
                    |
                    v
          HTML / CSS / JavaScript
```

The frontend collects clinical descriptors and sends them to a Python backend.

The backend handles preprocessing, model execution and result formatting.

### 8.2. Loading the Models

Example:

```python
import joblib

mass_preprocessor = joblib.load(
    "models/mass_preprocessor.joblib"
)

calc_preprocessor = joblib.load(
    "models/calc_preprocessor.joblib"
)

mass_lr = joblib.load(
    "models/mass_logistic_regression.joblib"
)

mass_rf = joblib.load(
    "models/mass_random_forest.joblib"
)

calc_lr = joblib.load(
    "models/calc_logistic_regression.joblib"
)

calc_rf = joblib.load(
    "models/calc_random_forest.joblib"
)
```

### 8.3. Expected Input Features

The backend must use the exact feature names and expected input types.

**Mass:**

```python
mass_features = [
    "breast_density",
    "left or right breast",
    "image view",
    "mass shape",
    "mass margins",
    "assessment",
    "subtlety"
]
```

**Calcification:**

```python
calc_features = [
    "breast_density",
    "left or right breast",
    "image view",
    "calc type",
    "calc distribution",
    "assessment",
    "subtlety"
]
```

The received input must be converted into a pandas DataFrame with the correct column names before being passed to the corresponding preprocessor.

Missing values and category normalization must follow the same conventions used during training.

### 8.4. Prediction Example

The following example demonstrates how to load a Mass preprocessing pipeline and Logistic Regression model, transform one input and generate a prediction.

```python
import pandas as pd
import joblib

pathology_mapping = {
    0: "BENIGN",
    1: "BENIGN_WITHOUT_CALLBACK",
    2: "MALIGNANT"
}

preprocessor = joblib.load(
    "models/mass_preprocessor.joblib"
)

model = joblib.load(
    "models/mass_logistic_regression.joblib"
)

input_data = pd.DataFrame([{
    "breast_density": 2,
    "left or right breast": "LEFT",
    "image view": "CC",
    "mass shape": "OVAL",
    "mass margins": "CIRCUMSCRIBED",
    "assessment": 3,
    "subtlety": 4
}])

X = preprocessor.transform(input_data)

prediction = model.predict(X)[0]

result = pathology_mapping[int(prediction)]

print("Predicted category:", result)
```

This is an illustrative example. The values must match valid categories and types from the dataset.

The Mass preprocessor must only be used with Mass models, and the Calcification preprocessor must only be used with Calcification models.

### 8.5. Other Application Resources

The application can also use:

| Resource | Potential Application Use |
|---|---|
| Association rules CSV files | Display discovered feature relationships |
| Selected rules CSV files | Explore the selected relationships |
| GraphML files | Visualize semantic knowledge graphs |
| Cluster assignments | Explore training cluster distributions |
| Evaluation results | Display model performance |
| Classification models | Generate experimental predictions |

**Clustering note:** The saved `.npy` files contain assignments for the training records in their original order. They are not a fitted K-Means model and cannot independently assign clusters to arbitrary new inputs.

---

## 9. Important Limitations

- The project uses clinical descriptors rather than direct mammography image analysis.
- Classification results are experimental and dataset-specific.
- Evaluation on the official CBIS-DDSM test split does not establish generalization to other hospitals or populations.
- Association rules represent statistical co-occurrences, not causal relationships.
- BI-RADS assessment is an input feature and must not be confused with the pathology target.
- Clusters are exploratory groups, not medical diagnoses.
- Predictions must not replace professional medical assessment.

---

## 10. Project Status

| Component | Status |
|---|---|
| Data Understanding | Completed |
| Data Preparation | Completed |
| Association Rule Mining | Completed |
| Clustering | Completed |
| Classification | Completed |
| Semantic Knowledge Graphs | Completed |
| Internal Model Evaluation | Completed |
| Official Test Evaluation | Completed |
| HTML/CSS/JavaScript Application | Pending |
| Backend API Integration | Pending |
| Final Documentation | In Progress |

---

## 11. Contributors

Academic project developed at **École Nationale d'Ingénieurs de Carthage (ENICarthage)**.

---

## 12. Disclaimer

This project is intended for educational and experimental purposes only.

It is not a clinically validated diagnostic tool. Its models and outputs must not be used as a substitute for professional medical diagnosis or clinical decision-making.