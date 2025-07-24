from pydantic import BaseModel


class Performance(BaseModel):
    method_name: str
    performance: float

    def __str__(self):
        return f"{self.method_name} ({self.performance:.2f}%)"
