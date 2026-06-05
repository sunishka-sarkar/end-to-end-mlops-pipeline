import joblib

MODEL_PATH = (
    "training/champion/champion_model.pkl"
)

model = joblib.load(
    MODEL_PATH
)