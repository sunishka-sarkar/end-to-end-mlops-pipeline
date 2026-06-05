import json
import subprocess

with open(
    "monitoring/drift_status.json"
) as f:

    drift_status = json.load(f)

print("\nDrift Status:")
print(drift_status)

if drift_status["drift_detected"]:

    print("\nDrift detected!")
    print("Starting retraining workflow...\n")

    subprocess.run(
        ["python", "retraining/retrain.py"]
    )

    subprocess.run(
        ["python", "retraining/evaluate.py"]
    )

    subprocess.run(
        ["python", "retraining/promote.py"]
    )

else:

    print(
        "\nNo drift detected."
    )