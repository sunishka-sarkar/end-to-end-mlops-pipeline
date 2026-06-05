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