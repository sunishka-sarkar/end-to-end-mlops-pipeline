import ast
import pandas as pd
from sqlalchemy import text

from database.db import engine


def load_production_data():

    with engine.connect() as conn:

        rows = conn.execute(
            text(
                """
                SELECT request_json
                FROM prediction_logs
                """
            )
        )

        records = []

        for row in rows:

            request_dict = ast.literal_eval(
                row[0]
            )

            records.append(
                request_dict
            )

    return pd.DataFrame(records)


if __name__ == "__main__":

    df = load_production_data()

    print(df.head())

    print("\nShape:")
    print(df.shape)