from pydantic import BaseModel


class Performance(BaseModel):
    performance: float = 0.0
    execution_seconds: int = 0
