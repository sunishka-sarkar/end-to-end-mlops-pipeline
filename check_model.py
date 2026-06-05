import joblib

preprocessor = joblib.load(
    "training/artifacts/preprocessor.pkl"
)

print(type(preprocessor))