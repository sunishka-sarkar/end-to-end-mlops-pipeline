import os
import json
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)

# =====================
# LOAD DATA
# =====================

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

# =====================
# TRAIN CHALLENGER
# =====================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

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

metrics = {
    "accuracy": float(
        accuracy_score(
            y_test,
            y_pred
        )
    ),
    "f1": float(
        f1_score(
            y_test,
            y_pred
        )
    ),
    "roc_auc": float(
        roc_auc_score(
            y_test,
            y_prob
        )
    )
}

# =====================
# SAVE CHALLENGER
# =====================

os.makedirs(
    "training/models",
    exist_ok=True
)

joblib.dump(
    model,
    "training/models/challenger_model.pkl"
)

with open(
    "training/models/challenger_metrics.json",
    "w"
) as f:
    json.dump(
        metrics,
        f,
        indent=4
    )

print("Challenger trained.")
print(metrics)