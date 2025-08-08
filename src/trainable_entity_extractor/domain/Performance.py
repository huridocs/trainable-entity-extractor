from pydantic import BaseModel


class Performance(BaseModel):
    method_name: str
    performance: float
    execution_seconds: int = 0

    @staticmethod
    def get_execution_time_string(execution_seconds: int):
        if execution_seconds <= 0:
            return "0s"

        if execution_seconds < 60:
            execution_time_string = f"{execution_seconds}s"
        elif execution_seconds < 3600:  # Less than 1 hour
            minutes = execution_seconds // 60
            seconds = execution_seconds % 60
            execution_time_string = f"{minutes}m {seconds}s"
        else:
            hours = execution_seconds // 3600
            remaining_seconds = execution_seconds % 3600
            minutes = remaining_seconds // 60
            execution_time_string = f"{hours}h {minutes}m"

        return execution_time_string

    def to_log(self, samples_count: int) -> str:
        mistakes = round(samples_count * (100 - self.performance) / 100)
        mistakes_text = f"{mistakes} mistakes" if mistakes != 1 else "1 mistake"
        return f"{self.method_name} - {self.get_execution_time_string(self.execution_seconds)} / {mistakes_text} / {self.performance:.2f}%"
