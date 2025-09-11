from pydantic import BaseModel


class DistributedPerformance(BaseModel):
    performance: float
    execution_seconds: int = 0
    shoul
