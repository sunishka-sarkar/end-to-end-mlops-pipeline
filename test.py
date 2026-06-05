import joblib

model = joblib.load(
    "training/champion/champion_model.pkl"
)

print(type(model))