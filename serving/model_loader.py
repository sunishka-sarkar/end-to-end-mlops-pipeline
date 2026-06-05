import joblib

MODEL_PATH = (
    "training/champion/champion_model.pkl"
)

PREPROCESSOR_PATH = (
    "training/artifacts/preprocessor.pkl"
)

model = joblib.load(
    MODEL_PATH
)

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)
print("Loading model from:", MODEL_PATH)
print("Loading preprocessor from:", PREPROCESSOR_PATH)

print("Model type:", type(model))
print("Preprocessor type:", type(preprocessor))