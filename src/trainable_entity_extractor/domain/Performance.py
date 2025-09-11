from pydantic import BaseModel


class Performance(BaseModel):
    performance: float = 0.0
    execution_seconds: int = 0
    should_be_retrained_with_more_data: bool = True
