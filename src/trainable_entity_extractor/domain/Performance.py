from pydantic import BaseModel


class Performance(BaseModel):
    method_name: str
    performance: float
    execution_seconds: int = 0

    def to_log(self, samples_count: int) -> str:
        mistakes = round(samples_count * (100 - self.performance) / 100)
        mistakes_text = f"{mistakes} mistakes" if mistakes != 1 else "1 mistake"
        text = f"{self.method_name} - {self.get_execution_time_string()} / {mistakes_text} / {self.performance:.2f}%"
        return text

    def get_execution_time_string(self):
        if self.execution_seconds <= 0:
            return "0s"

        if self.execution_seconds < 60:
            execution_time_string = f"{self.execution_seconds}s"
        elif self.execution_seconds < 3600:  # Less than 1 hour
            minutes = self.execution_seconds // 60
            seconds = self.execution_seconds % 60
            execution_time_string = f"{minutes}m {seconds}s"
        else:
            hours = self.execution_seconds // 3600
            remaining_seconds = self.execution_seconds % 3600
            minutes = remaining_seconds // 60
            execution_time_string = f"{hours}h {minutes}m"
        return execution_time_string
