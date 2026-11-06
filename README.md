# 🏭 WaferGuard AI — Semiconductor Wafer Quality Prediction & Fault Detection

**An ML-powered approach to Industrial Intelligence and Smart Manufacturing**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![Status](https://img.shields.io/badge/Status-Prototype-success)
![Challenge](https://img.shields.io/badge/DATA%20DIVERGENCE-First%20Place-gold)

## 🏆 Achievement

**1st Place — Track 3: Advanced Level**

DATA DIVERGENCE | 24-Hour Machine Learning Challenge  
Organized by DEVS REC

---

## 📌 Project Overview

WaferGuard AI is a machine learning-based semiconductor wafer quality prediction and fault detection system developed to identify potentially defective wafers using manufacturing sensor measurements.

Semiconductor manufacturing involves complex processes where hundreds of sensor measurements are collected for each wafer. Analysing these measurements manually is challenging, particularly when the dataset contains missing values, redundant features, and highly imbalanced target classes.

WaferGuard AI addresses this challenge by implementing a complete machine learning pipeline that processes industrial sensor data, identifies meaningful patterns, and predicts whether a wafer is likely to pass or fail quality inspection.

The system also includes a reusable inference pipeline and a Streamlit application prototype for interactive predictions.

## 🎯 Problem Statement

Develop a machine learning model that analyses semiconductor manufacturing sensor data and predicts wafer quality.

The primary objectives are:

- Identify patterns associated with defective wafers.
- Handle missing and inconsistent sensor measurements.
- Address class imbalance in manufacturing data.
- Select relevant sensor features.
- Train and evaluate multiple classification algorithms.
- Generate reliable predictions on previously unseen data.

## 📊 Dataset Description

The project uses the competition-provided `train.csv`, based on the SECOM semiconductor manufacturing dataset.

| Dataset attribute | Value |
|---|---:|
| Total wafer records | 1,253 |
| Sensor features | 590 |
| Good wafers (Class 0) | 1,170 |
| Defective wafers (Class 1) | 83 |
| Missing sensor values | 33,940 |
| Constant sensor features | 116 |

**Target variable:**
- `0` — PASS / Good wafer
- `1` — FAIL / Defective wafer

The dataset is highly imbalanced, with defective wafers representing approximately 6.62% of the records.

This makes minority-class detection particularly important.

## ⚙️ Technical Workflow

The project follows an end-to-end machine learning workflow.

### 1. Exploratory Data Analysis (EDA)

Performed initial analysis to understand the manufacturing dataset.

- Dataset dimensions and feature inspection.
- Missing-value distribution.
- Target class distribution.
- Constant feature identification.
- Sensor distribution analysis.
- Chronological data exploration.

### 2. Data Preprocessing

Implemented preprocessing techniques using Scikit-learn.

| Technique | Purpose |
|---|---|
| Median Imputation | Handles missing sensor readings |
| VarianceThreshold | Removes constant features |
| RobustScaler | Scales features using median and interquartile range |
| SelectKBest | Selects the 100 most relevant features |

Preprocessing was implemented within Scikit-learn pipelines to ensure that transformations are fitted on training data during cross-validation.

### 3. Handling Class Imbalance

Since defective wafers are significantly underrepresented, we used:

`class_weight="balanced_subsample"`

This adjusts class contributions during Random Forest training to improve minority-class learning.

### 4. Machine Learning Models

Multiple classification algorithms were experimented with and compared.

| Algorithm | Application |
|---|---|
| Logistic Regression | Baseline classification |
| Random Forest | Ensemble classification |
| Extra Trees | Randomized tree ensemble |
| HistGradientBoosting | Gradient boosting classification |

### 5. Hyperparameter Optimization

RandomizedSearchCV was used to explore Random Forest configurations.

The selected configuration:

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=8,
    max_features=0.4,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42
)
```

The tuning process reported a cross-validation Average Precision of **0.2349**.

### 6. Model Evaluation

The model was evaluated using multiple validation approaches.

- Stratified 5-fold cross-validation.
- Out-of-fold prediction evaluation.
- Chronological 80:20 validation.

**Evaluation metrics:**

- Average Precision (AP)
- Balanced Accuracy
- Confusion Matrix
- Failure Detection Rate

These metrics provide a more meaningful evaluation than ordinary accuracy for an imbalanced classification problem.

## 📈 Experimental Results

### Random Forest — Out-of-Fold Results

| Metric | Result |
|---|---:|
| Average Precision | 0.2027 |
| Balanced Accuracy | 0.6707 |
| Probability Threshold | 0.15 |
| Defective wafers detected | 57 / 83 |
| False positives | 404 |
| False negatives | 26 |

### Confusion Matrix

| | Predicted PASS | Predicted FAIL |
|---|---:|---:|
| Actual PASS | 766 | 404 |
| Actual FAIL | 26 | 57 |

### Chronological Validation Comparison

At a probability threshold of 0.15:

| Metric | Random Forest | HistGradientBoosting |
|---|---:|---:|
| Average Precision | 0.1556 | 0.2130 |
| Balanced Accuracy | 0.6552 | 0.5874 |
| Failures detected | 12/18 | 5/18 |
| False alarms | 83 | 24 |

Random Forest was retained as the primary candidate because it detected more actual failures in this chronological comparison.

**Note:** These are internal validation results, not competition hidden-test results. The selected probability threshold is provisional and involves a trade-off between failure detection and false alarms.

## 🧠 System Architecture

```text
             Industrial Sensor Dataset
                       |
                       v
                Data Exploration
                       |
                       v
                Data Preprocessing
                       |
              +--------+--------+
              |                 |
              v                 v
         Median Imputation   Variance Filtering
              |                 |
              +--------+--------+
                       |
                       v
                  RobustScaler
                       |
                       v
                  SelectKBest
                  (100 Features)
                       |
                       v
               Tuned Random Forest
                       |
                       v
              Failure Probability
                       |
                       v
              Threshold = 0.15
                       |
                +------+------+
                |             |
                v             v
             PASS (0)      FAIL (1)
                            |
                            v
                    Further Inspection
```

## 💻 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| Google Colab | Development and experimentation |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Preprocessing, model training and evaluation |
| Random Forest | Primary classification model |
| Joblib | Model serialization |
| Streamlit | Interactive application prototype |
| GitHub | Version control and project hosting |

## 🚀 Application Features

The Streamlit prototype provides an interface for demonstrating wafer quality predictions.

**Key functionality:**

- Upload manufacturing sensor data in CSV format.
- Validate expected sensor columns.
- Process data using the saved ML pipeline.
- Generate failure probabilities.
- Classify wafers as PASS or FAIL.
- View prediction results.
- Download prediction outputs.

The inference pipeline is implemented separately to allow model reuse without retraining.

## 📁 Project Structure

```text
WaferGuard_AI/
│
├── waferguard_training.ipynb
│   └── EDA, preprocessing, experiments,
│       model training and evaluation
│
├── waferguard_model_final.joblib
│   └── Saved trained ML pipeline
│
├── inference.py
│   └── Reusable prediction function
│
├── app.py
│   └── Streamlit application
│
├── methodology_summary.pdf
│   └── Project methodology and evaluation
│
├── requirements.txt
│   └── Python dependencies
│
└── README.md
    └── Project documentation
```

## 🛠️ Installation and Execution

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd WaferGuard_AI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application

```bash
streamlit run app.py
```

### 4. Use the Application

1. Open the Streamlit application in your browser.
2. Upload a CSV containing the expected sensor features.
3. Run the prediction workflow.
4. View the predicted failure probabilities and PASS/FAIL classifications.
5. Download the prediction results.

**Note:** The competition dataset is not included in this repository. Obtain the authorized dataset separately. The saved model artifact must be present to run inference.

## 🔬 Additional Experiments

We also explored:

- Missingness indicators to investigate whether missing sensor patterns improve prediction.
- Statistical feature selection.
- Sensor distribution shifts between earlier and later observations.
- The effect of chronological validation on model performance.

Missingness indicators provided only a small Average Precision improvement and reduced Balanced Accuracy in the tested comparison, so they were not retained in the final pipeline.

These experiments helped guide model selection using measured performance rather than assumptions.

## ⚠️ Limitations and Future Scope

The current implementation is a machine learning prototype, not a production-integrated semiconductor inspection system.

Potential improvements include:

- Advanced feature interaction analysis.
- Temporal drift detection.
- Probability calibration and threshold optimization.
- Additional ensemble and boosting algorithms.
- Independent hidden-test evaluation.
- Real-time manufacturing sensor integration.
- Production deployment and monitoring.

A FAIL prediction is an inspection flag, not definitive proof of a physical wafer defect.

## 👥 Team

**Team Vortex**

Developed as part of DATA DIVERGENCE — DEVS REC.

## 🙌 Acknowledgements

We thank **DEVS REC** and the faculty coordinators for organizing DATA DIVERGENCE and providing an opportunity to work on a practical machine learning challenge in Industrial Intelligence and Smart Manufacturing.

The 24-hour challenge provided valuable experience in data analysis, machine learning experimentation, model optimization, debugging, and collaborative problem-solving.

---

**WaferGuard AI — Turning Industrial Sensor Data into Intelligent Quality Predictions.** 🚀
