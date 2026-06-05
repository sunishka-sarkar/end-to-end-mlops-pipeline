from database.models import PredictionLog


def save_prediction(
    db,
    prediction,
    probability,
    request_json
):

    row = PredictionLog(
        prediction=prediction,
        probability=probability,
        model_version="v1",
        request_json=request_json
    )

    db.add(row)
    db.commit()