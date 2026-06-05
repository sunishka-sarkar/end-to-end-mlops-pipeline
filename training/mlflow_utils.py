import mlflow

mlflow.set_tracking_uri(
    "file:./mlruns"
)

mlflow.set_experiment(
    "Customer_Churn_Models"
)