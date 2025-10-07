from pydantic import BaseModel


class Performance(BaseModel):
    method_name: str = "Unknown Method"
    performance: float = 0.0
    execution_seconds: int = 0
    is_perfect: bool = False
    failed: bool = False
    testing_samples_count: int = 0
    training_samples_count: int = 0
