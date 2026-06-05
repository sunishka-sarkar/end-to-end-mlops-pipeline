import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(
    "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["Churn"] = df["Churn"].map(
    {
        "Yes": 1,
        "No": 0
    }
)

# ==========================
# FEATURES
# ==========================

X = df.drop(
    columns=["customerID", "Churn"]
)

y = df["Churn"]

# ==========================
# SPLIT
# ==========================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
)

# ==========================
# FEATURE LISTS
# ==========================

numerical_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    col for col in X.columns
    if col not in numerical_features
]

# ==========================
# PREPROCESSOR
# ==========================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])

preprocessor = ColumnTransformer([
    (
        "num",
        numeric_pipeline,
        numerical_features
    ),
    (
        "cat",
        categorical_pipeline,
        categorical_features
    )
])

# ==========================
# FULL PIPELINE
# ==========================

full_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

# ==========================
# TRAIN
# ==========================

full_pipeline.fit(
    X_train,
    y_train
)

# ==========================
# SAVE
# ==========================

os.makedirs(
    "training/champion",
    exist_ok=True
)

joblib.dump(
    full_pipeline,
    "training/champion/champion_model.pkl"
)

print(
    "Champion Pipeline Saved!"
)