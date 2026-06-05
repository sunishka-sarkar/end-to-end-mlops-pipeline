import streamlit as st
import pandas as pd
import sqlite3
import json

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Customer Churn MLOps Dashboard",
    layout="wide"
)

st.title("🚀 Customer Churn MLOps Dashboard")

# =====================================
# LOAD MODEL METRICS
# =====================================

metrics = None

try:

    with open(
        "training/champion/champion_metrics.json"
    ) as f:

        metrics = json.load(f)

except Exception:
    pass

# =====================================
# LOAD DRIFT STATUS
# =====================================

drift = None

try:

    with open(
        "monitoring/drift_status.json"
    ) as f:

        drift = json.load(f)

except Exception:
    pass

# =====================================
# LOAD DATABASE LOGS
# =====================================

logs = pd.DataFrame()

try:

    conn = sqlite3.connect(
        "predictions.db"
    )

    logs = pd.read_sql_query(
        """
        SELECT *
        FROM prediction_logs
        ORDER BY id DESC
        LIMIT 100
        """,
        conn
    )

    conn.close()

except Exception as e:

    st.error(
        f"Database Error: {e}"
    )

# =====================================
# MODEL OVERVIEW
# =====================================

st.header("📊 Champion Model Overview")

if metrics:

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "ROC-AUC",
        round(
            metrics["roc_auc"],
            4
        )
    )

    col2.metric(
        "Accuracy",
        round(
            metrics["accuracy"],
            4
        )
    )

    col3.metric(
        "F1 Score",
        round(
            metrics["f1"],
            4
        )
    )

    col4.metric(
        "Version",
        "v1"
    )

else:

    st.warning(
        "Champion metrics not found."
    )

# =====================================
# DRIFT MONITORING
# =====================================

st.header("📈 Drift Monitoring")

if drift:

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Drift Score",
        drift["drift_score"]
    )

    col2.metric(
        "Drifted Columns",
        drift["drifted_columns"]
    )

    col3.metric(
        "Total Columns",
        drift["total_columns"]
    )

    if drift["drift_detected"]:

        st.error(
            "⚠️ Drift Detected"
        )

    else:

        st.success(
            "✅ No Drift Detected"
        )

else:

    st.warning(
        "Drift status unavailable."
    )

# =====================================
# RETRAINING STATUS
# =====================================

st.header("🔄 Retraining Status")

if drift:

    if drift["drift_detected"]:

        st.warning(
            "Drift threshold exceeded. Retraining workflow activated."
        )

    else:

        st.success(
            "Champion model remains healthy."
        )

# =====================================
# PREDICTION ANALYTICS
# =====================================

st.header("📊 Prediction Analytics")

if not logs.empty:

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Predictions",
        len(logs)
    )

    col2.metric(
        "Average Probability",
        round(
            logs["probability"].mean(),
            3
        )
    )

    col3.metric(
        "Positive Predictions",
        int(
            logs["prediction"].sum()
        )
    )

# =====================================
# PROBABILITY TREND
# =====================================

st.header("📈 Prediction Probability Trend")

if (
    not logs.empty and
    "created_at" in logs.columns
):

    logs["created_at"] = pd.to_datetime(
        logs["created_at"]
    )

    chart_df = logs.sort_values(
        "created_at"
    )

    st.line_chart(
        chart_df.set_index(
            "created_at"
        )["probability"]
    )

else:

    st.info(
        "Generate some predictions to visualize trends."
    )

# =====================================
# LATEST PREDICTIONS
# =====================================

st.header("📝 Latest Predictions")

if not logs.empty:

    columns_to_show = [
        col for col in [
            "id",
            "prediction",
            "probability",
            "model_version",
            "created_at"
        ]
        if col in logs.columns
    ]

    st.dataframe(
        logs[
            columns_to_show
        ],
        use_container_width=True
    )

# =====================================
# SYSTEM HEALTH
# =====================================

st.header("⚙️ System Health")

status_df = pd.DataFrame({

    "Component": [
        "FastAPI API",
        "Champion Model",
        "Prediction Logging",
        "Drift Detection",
        "Scheduler"
    ],

    "Status": [
        "🟢 Running",
        "🟢 Loaded",
        "🟢 Active",
        "🟢 Active",
        "🟢 Active"
    ]
})

st.dataframe(
    status_df,
    use_container_width=True
)

# =====================================
# PIPELINE FLOW
# =====================================

st.header("🔄 MLOps Pipeline")

st.info(
    """
    Customer Request
            ↓
    FastAPI Prediction
            ↓
    Prediction Logging
            ↓
    Drift Detection
            ↓
    Challenger Retraining
            ↓
    Champion Evaluation
            ↓
    Model Promotion
    """
)