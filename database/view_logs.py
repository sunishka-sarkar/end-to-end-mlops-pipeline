from sqlalchemy import text
from database.db import engine

with engine.connect() as conn:

    rows = conn.execute(
        text(
            """
            SELECT
                id,
                prediction,
                probability,
                model_version,
                request_json
            FROM prediction_logs
            """
        )
    )

    for row in rows:
        print(row)