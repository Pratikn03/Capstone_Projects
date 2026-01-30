from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

class AnomalyResponse(BaseModel):
    anomalies: List[bool] = []

@router.get("", response_model=AnomalyResponse)
def get_anomalies():
    # TODO: wire anomaly detection
    return AnomalyResponse(anomalies=[])
