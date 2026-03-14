# Lab 2: Wine Quality Prediction with MLflow Experiment Tracking

This lab demonstrates the full machine learning lifecycle using MLflow — from data preprocessing and model training to model registration, deployment, and real-time inference. It uses the Wine Quality dataset and covers experiment tracking, model versioning, hyperparameter tuning, and batch/real-time inference.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Project Structure](#project-structure)
- [Lab Walkthrough](#lab-walkthrough)
  - [Step 1–3: Data Loading and Preprocessing](#steps-1-3-data-loading-and-preprocessing)
  - [Step 4–7: Exploratory Data Analysis](#steps-4-7-exploratory-data-analysis)
  - [Step 8: Data Splitting](#step-8-data-splitting)
  - [Step 9–10: Baseline Random Forest Model](#steps-9-10-baseline-random-forest-model)
  - [Step 11–13: Model Registration and Production Promotion](#steps-11-13-model-registration-and-production-promotion)
  - [Step 15–16: XGBoost with Hyperopt Tuning](#steps-15-16-xgboost-with-hyperopt-tuning)
  - [Step 17–18: Update Production Model](#steps-17-18-update-production-model)
  - [Step 19: Batch Inference](#step-19-batch-inference)
  - [Step 20: Real-Time Inference](#step-20-real-time-inference)
- [New Features Added](#new-features-added)
  - [Feature 1: LightGBM Model](#feature-1-lightgbm-model)
  - [Feature 2: Logistic Regression Model](#feature-2-logistic-regression-model)
  - [Feature 3: Extended Metrics Logging](#feature-3-extended-metrics-logging)
  - [Feature 4: Confusion Matrix Artifact Logging](#feature-4-confusion-matrix-artifact-logging)
- [Model Comparison Results](#model-comparison-results)
- [MLflow UI](#mlflow-ui)

---

## Overview

This lab covers:
- Loading and preprocessing red and white wine datasets
- Binary classification: predicting whether a wine is "high quality" (quality score ≥ 7)
- Training and tracking multiple models with MLflow
- Hyperparameter tuning using Hyperopt
- Model registration and stage management in MLflow Model Registry
- Batch and real-time inference using the deployed production model
- **4 new features** extending the original lab

---

## Prerequisites

- Python 3.10+
- Java (for PySpark — optional, can be skipped)
- Git

---

## Setup

**1. Clone the repository and navigate to Lab 2:**
```bash
cd Labs/Experiment_Tracking_Labs/Mlflow_Labs/Lab2
```

**2. Create and activate a virtual environment:**
```bash
python -m venv .venv

# Mac/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
pip install lightgbm
```

**4. Register the Jupyter kernel:**
```bash
pip install ipykernel
python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"
```

**5. Launch Jupyter:**
```bash
python -m jupyter notebook
```

Open `wine_quality_lab2.ipynb` and select kernel **Python (.venv)**.

---

## Project Structure

```
Lab2/
├── data/
│   ├── winequality-red.csv
│   ├── winequality-white.csv
│   └── winequality.names
├── wine_quality_lab2.ipynb
├── requirements.txt
└── README.md
```

---

## Lab Walkthrough

### Steps 1-3: Data Loading and Preprocessing

Load red and white wine datasets and combine them with an `is_red` indicator variable.

```python
import pandas as pd

white_wine = pd.read_csv("data/winequality-white.csv", sep=";")
red_wine = pd.read_csv("data/winequality-red.csv", sep=",")

red_wine['is_red'] = 1
white_wine['is_red'] = 0

data = pd.concat([red_wine, white_wine], axis=0)
data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)
```

### Steps 4-7: Exploratory Data Analysis

Visualize the quality distribution and identify key predictors via box plots. Key findings:
- **Alcohol** is positively correlated with quality
- **Density** is negatively correlated with quality

Quality is binarized: wines with score ≥ 7 are labeled as high quality (1), others as 0.

```python
high_quality = (data.quality >= 7).astype(int)
data.quality = high_quality
```

### Step 8: Data Splitting

Data is split into 60% training, 20% validation, and 20% test sets.

```python
from sklearn.model_selection import train_test_split

X = data.drop(["quality"], axis=1)
y = data.quality

X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=123)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123)
```

### Steps 9-10: Baseline Random Forest Model

A Random Forest classifier is trained and logged to MLflow with AUC as the evaluation metric. A custom `SklearnModelWrapper` is used to return class probabilities via `predict_proba`.

```python
with mlflow.start_run(run_name='untuned_random_forest'):
    model = RandomForestClassifier(n_estimators=10, random_state=np.random.RandomState(123))
    model.fit(X_train, y_train)
    
    auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    mlflow.log_param('n_estimators', 10)
    mlflow.log_metric('auc', auc_score)
```

**Baseline AUC: ~0.854**

### Steps 11-13: Model Registration and Production Promotion

The trained model is registered in the MLflow Model Registry and promoted to the Production stage.

```python
model_version = mlflow.register_model(f"runs:/{run_id}/random_forest_model", "wine_quality")

client.transition_model_version_stage(
    name="wine_quality",
    version=model_version.version,
    stage="Production",
)
```

### Steps 15-16: XGBoost with Hyperopt Tuning

XGBoost models are tuned using Bayesian optimization (TPE algorithm) via Hyperopt. Each trial is logged as a **nested run** under the parent run `xgboost_models`.

```python
with mlflow.start_run(run_name='xgboost_models'):
    best_params = fmin(
        fn=train_model,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials(),
    )
```

### Steps 17-18: Update Production Model

The best XGBoost run is registered and promoted to Production, while the old Random Forest version is archived.

### Step 19: Batch Inference

The production model is loaded and used to run batch predictions on the test set.

```python
model = mlflow.pyfunc.load_model(f"models:/wine_quality/production")
batch_predictions = model.predict(X_test)
print(batch_predictions[:10])
```

### Step 20: Real-Time Inference

The model is served via MLflow's built-in REST API server. Run this in a separate terminal:

```bash
mlflow models serve --env-manager=local -m models:/wine_quality/production -h 0.0.0.0 -p 5001
```

Then send predictions via HTTP POST:

```python
import requests

url = 'http://localhost:5001/invocations'
data_dict = {"dataframe_split": X_test.to_dict(orient='split')}
response = requests.post(url, json=data_dict)
print(response.json())
```

---

## New Features Added

### Feature 1: LightGBM Model

A LightGBM classifier is added as a third model alongside Random Forest and XGBoost. LightGBM uses gradient boosting with histogram-based tree learning, making it faster and more memory-efficient than XGBoost on larger datasets.

```python
import lightgbm as lgb

with mlflow.start_run(run_name='lightgbm_model'):
    lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=123)
    lgb_model.fit(X_train, y_train)
    
    mlflow.log_param('n_estimators', 100)
    mlflow.log_param('model_type', 'lightgbm')
    mlflow.log_metric('auc', roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1]))
    mlflow.sklearn.log_model(lgb_model, "lightgbm_model")
```

**LightGBM AUC: ~0.898**

---

### Feature 2: Logistic Regression Model

A Logistic Regression model is added as a linear baseline for comparison. This provides interpretable coefficients and helps demonstrate the performance gap between linear and tree-based approaches on this dataset.

```python
from sklearn.linear_model import LogisticRegression

with mlflow.start_run(run_name='logistic_regression_model'):
    lr_model = LogisticRegression(max_iter=1000, random_state=123)
    lr_model.fit(X_train, y_train)
    
    mlflow.log_param('max_iter', 1000)
    mlflow.log_param('model_type', 'logistic_regression')
    mlflow.log_metric('auc', roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1]))
    mlflow.sklearn.log_model(lr_model, "logistic_regression_model")
```

**Logistic Regression AUC: ~0.812**

---

### Feature 3: Extended Metrics Logging

Beyond AUC, precision, recall, and F1 score are logged for all new models. AUC alone can be misleading on imbalanced datasets — these additional metrics provide a fuller picture of model performance.

```python
from sklearn.metrics import precision_score, recall_score, f1_score

mlflow.log_metric('precision', precision_score(y_test, preds))
mlflow.log_metric('recall', recall_score(y_test, preds))
mlflow.log_metric('f1', f1_score(y_test, preds))
```

These metrics are visible in the MLflow UI under each run, enabling direct comparison across models.

---

### Feature 4: Confusion Matrix Artifact Logging

A confusion matrix plot is generated for each new model and saved as an artifact in MLflow using `mlflow.log_figure()`. This allows visual inspection of false positives and false negatives directly from the MLflow UI.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)
ax.set_title("LightGBM Confusion Matrix")
mlflow.log_figure(fig, "confusion_matrix_lightgbm.png")
plt.close()
```

The saved confusion matrix PNGs are accessible under the **Artifacts** tab of each run in the MLflow UI.

---

## Model Comparison Results

After running all models, the final comparison table (sorted by AUC):

| Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| XGBoost (tuned) | ~0.920 | — | — | — |
| LightGBM | 0.898 | 0.745 | 0.559 | 0.638 |
| Random Forest | ~0.854 | — | — | — |
| Logistic Regression | 0.812 | 0.663 | 0.223 | 0.333 |

Tree-based models (XGBoost, LightGBM, Random Forest) significantly outperform the linear Logistic Regression model on this dataset, which has non-linear feature interactions.

---

## MLflow UI

To view all experiments, runs, metrics, parameters, and artifacts in a browser dashboard, run in a separate terminal:

```bash
mlflow ui --port=5001
```

Then open `http://localhost:5001` in your browser.

You can:
- Compare all model runs side by side
- View logged parameters and metrics
- Download confusion matrix artifacts from the Artifacts tab
- See nested runs from the XGBoost hyperparameter sweep
