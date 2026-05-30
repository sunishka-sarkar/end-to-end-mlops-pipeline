import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

print("Dataset Shape:", df.shape)

# ==========================================
# FIX TOTALCHARGES
# ==========================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("Missing TotalCharges:",
      df["TotalCharges"].isnull().sum())

# ==========================================
# TARGET ENCODING
# ==========================================

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# ==========================================
# FEATURE LISTS
# ==========================================

numerical_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    col for col in df.columns
    if col not in numerical_features
    and col not in ["customerID", "Churn"]
]

# ==========================================
# FEATURES AND TARGET
# ==========================================

X = df.drop(
    columns=["customerID", "Churn"]
)

y = df["Churn"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# ==========================================
# NUMERIC PIPELINE
# ==========================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# ==========================================
# CATEGORICAL PIPELINE
# ==========================================

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# ==========================================
# PREPROCESSOR
# ==========================================

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

print("Preprocessor Created")

# ==========================================
# TRANSFORM DATA
# ==========================================

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)

print("Processed Train Shape:",
      X_train_processed.shape)

print("Processed Test Shape:",
      X_test_processed.shape)

print("Train Type:",
      type(X_train_processed))

# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

# ==========================================
# SAVE FEATURES
# ==========================================

pd.DataFrame(
    X_train_processed
).to_parquet(
    "data/processed/train_features.parquet",
    index=False
)

pd.DataFrame(
    X_test_processed
).to_parquet(
    "data/processed/test_features.parquet",
    index=False
)

# ==========================================
# SAVE LABELS
# ==========================================

pd.DataFrame(
    y_train
).to_parquet(
    "data/processed/y_train.parquet",
    index=False
)

pd.DataFrame(
    y_test
).to_parquet(
    "data/processed/y_test.parquet",
    index=False
)

print("\nFILES SAVED SUCCESSFULLY")
print("train_features.parquet")
print("test_features.parquet")
print("y_train.parquet")
print("y_test.parquet")