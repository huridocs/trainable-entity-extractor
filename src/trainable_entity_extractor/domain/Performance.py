from pydantic import BaseModel


class Performance(BaseModel):
    method_name: str
    performance: float

    def to_log(self, samples_count: int) -> str:
        return f"{self.method_name} - {round(samples_count * (100 - self.performance) / 100)} mistakes / {self.performance:.2f}%"
