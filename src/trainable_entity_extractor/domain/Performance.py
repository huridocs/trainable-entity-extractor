from pydantic import BaseModel


class Performance(BaseModel):
    method_name: str
    performance: float

    def to_log(self, samples_count: int) -> str:
        mistakes = round(samples_count * (100 - self.performance) / 100)
        mistakes_text = f"{mistakes} mistakes" if mistakes != 1 else "1 mistake"
        text = f"{self.method_name} - {mistakes_text} / {self.performance:.2f}%"
        return text
