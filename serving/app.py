from fastapi import FastAPI
import pandas as pd

from database.db import SessionLocal
from database.crud import save_prediction

from serving.schemas import CustomerRequest
from serving.model_loader import (
    model,
    preprocessor
)

app = FastAPI(
    title="Customer Churn Prediction API"
)


@app.get("/")
def home():
    return {
        "message": "Customer Churn API Running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: CustomerRequest):

    request_data = request.model_dump()

    # Raw customer data
    data = pd.DataFrame(
        [request_data]
    )

    # Apply preprocessing
    processed_data = preprocessor.transform(
        data
    )
    print("RAW COLUMNS:")
    print(data.columns.tolist())

    print("PROCESSED SHAPE:")
    print(processed_data.shape)

    # Predict
    prediction = model.predict(
        processed_data
    )[0]

    probability = model.predict_proba(
        processed_data
    )[0][1]

    # Save prediction
    db = SessionLocal()

    try:

        save_prediction(
            db=db,
            prediction=int(prediction),
            probability=float(probability),
            request_json=str(request_data)
        )

    finally:
        db.close()

    return {
        "prediction": int(prediction),
        "probability": round(
            float(probability),
            4
        )
    }