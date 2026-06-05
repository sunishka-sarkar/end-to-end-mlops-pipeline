import time
import subprocess
from datetime import datetime

CHECK_INTERVAL = 60


def log(message):
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    print(
        f"[{timestamp}] {message}"
    )


log("=" * 60)
log("MLOps Scheduler Started")
log(f"Check Interval: {CHECK_INTERVAL} seconds")
log("=" * 60)

while True:

    try:

        log("Starting Drift Detection")

        drift_result = subprocess.run(
            ["python", "-m", "monitoring.drift_report"],
            capture_output=True,
            text=True
        )

        if drift_result.returncode == 0:

            log("Drift Detection Completed Successfully")

        else:

            log("Drift Detection Failed")
            print(drift_result.stderr)

        log("Starting Retraining Pipeline")

        pipeline_result = subprocess.run(
            ["python", "orchestration/run_pipeline.py"],
            capture_output=True,
            text=True
        )

        if pipeline_result.returncode == 0:

            log("Retraining Pipeline Completed")

        else:

            log("Retraining Pipeline Failed")
            print(pipeline_result.stderr)

        log(
            f"Sleeping for {CHECK_INTERVAL} seconds"
        )

        print(
            "\n" + "-" * 60 + "\n"
        )

        time.sleep(
            CHECK_INTERVAL
        )

    except KeyboardInterrupt:

        log(
            "Scheduler Stopped By User"
        )

        break

    except Exception as e:

        log(
            f"Unexpected Error: {str(e)}"
        )

        time.sleep(
            CHECK_INTERVAL
        )