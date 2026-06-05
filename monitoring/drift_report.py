import json
import pandas as pd

from monitoring.production_data import (
    load_production_data
)

# ==========================
# TRAINING DATA
# ==========================

reference_df = pd.read_csv(
    "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

reference_df["TotalCharges"] = pd.to_numeric(
    reference_df["TotalCharges"],
    errors="coerce"
)

reference_df = reference_df.drop(
    columns=["customerID", "Churn"]
)

# ==========================
# PRODUCTION DATA
# ==========================

current_df = load_production_data()

print("Reference Shape:", reference_df.shape)
print("Current Shape:", current_df.shape)

# ==========================
# SIMPLE DRIFT CHECK
# ==========================

numeric_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

drifted_columns = 0

for col in numeric_cols:

    ref_mean = reference_df[col].mean()
    cur_mean = current_df[col].mean()

    diff_percent = abs(
        cur_mean - ref_mean
    ) / ref_mean

    print(
        f"{col}: "
        f"reference={ref_mean:.2f}, "
        f"current={cur_mean:.2f}, "
        f"diff={diff_percent:.2f}"
    )

    if diff_percent > 0.20:
        drifted_columns += 1

drift_score = (
    drifted_columns /
    len(numeric_cols)
)

drift_detected = (
    drift_score > 0.30
)

# ==========================
# SAVE STATUS
# ==========================

status = {
    "drift_detected": drift_detected,
    "drift_score": round(
        drift_score,
        4
    ),
    "drifted_columns": drifted_columns,
    "total_columns": len(
        numeric_cols
    )
}

with open(
    "monitoring/drift_status.json",
    "w"
) as f:

    json.dump(
        status,
        f,
        indent=4
    )

print("\n====================")
print("DRIFT RESULTS")
print("====================")
print(status)