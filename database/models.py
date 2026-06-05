from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    Text,
    DateTime
)

from datetime import datetime

from database.db import Base


class PredictionLog(Base):

    __tablename__ = "prediction_logs"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    prediction = Column(
        Integer
    )

    probability = Column(
        Float
    )

    model_version = Column(
        String
    )

    request_json = Column(
        Text
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )