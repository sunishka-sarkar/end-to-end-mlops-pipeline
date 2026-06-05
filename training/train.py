import os
import joblib
import pandas as pd
import mlflow

from evaluate import evaluate_model
from mlflow_utils import *

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ==========================================
# LOAD FEATURE STORE
# ==========================================

X_train = pd.read_parquet(
    "data/processed/train_features.parquet"
)

X_test = pd.read_parquet(
    "data/processed/test_features.parquet"
)

y_train = pd.read_parquet(
    "data/processed/y_train.parquet"
).squeeze()

y_test = pd.read_parquet(
    "data/processed/y_test.parquet"
).squeeze()

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# ==========================================
# TRAIN MODEL
# ==========================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

print("\nTraining Logistic Regression...")

with mlflow.start_run(
    run_name="Logistic_Regression"
):

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    metrics = evaluate_model(
        y_test,
        y_pred,
        y_prob
    )

    mlflow.log_param(
        "model_type",
        "LogisticRegression"
    )

    mlflow.log_param(
        "max_iter",
        1000
    )

    mlflow.log_metrics(
        metrics
    )

    print(metrics)

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================
# EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("\n===== METRICS =====")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")

# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs(
    "training/models",
    exist_ok=True
)

joblib.dump(
    model,
    "training/champion/champion_model.pkl"
)

print(
    "\nChampion Model Updated!"
)

print(
    "training/models/logistic_regression.pkl"
)
print("\n==============================")
print("RANDOM FOREST")
print("==============================")

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(
    X_test
)

rf_prob = rf_model.predict_proba(
    X_test
)[:, 1]

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

rf_precision = precision_score(
    y_test,
    rf_pred
)

rf_recall = recall_score(
    y_test,
    rf_pred
)

rf_f1 = f1_score(
    y_test,
    rf_pred
)

rf_auc = roc_auc_score(
    y_test,
    rf_prob
)

print(f"Accuracy : {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall   : {rf_recall:.4f}")
print(f"F1 Score : {rf_f1:.4f}")
print(f"ROC AUC  : {rf_auc:.4f}")
print("\n==============================")
print("XGBOOST")
print("==============================")

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(
    X_train,
    y_train
)

xgb_pred = xgb_model.predict(
    X_test
)

xgb_prob = xgb_model.predict_proba(
    X_test
)[:, 1]

xgb_accuracy = accuracy_score(
    y_test,
    xgb_pred
)

xgb_precision = precision_score(
    y_test,
    xgb_pred
)

xgb_recall = recall_score(
    y_test,
    xgb_pred
)

xgb_f1 = f1_score(
    y_test,
    xgb_pred
)

xgb_auc = roc_auc_score(
    y_test,
    xgb_prob
)

print(f"Accuracy : {xgb_accuracy:.4f}")
print(f"Precision: {xgb_precision:.4f}")
print(f"Recall   : {xgb_recall:.4f}")
print(f"F1 Score : {xgb_f1:.4f}")
print(f"ROC AUC  : {xgb_auc:.4f}")