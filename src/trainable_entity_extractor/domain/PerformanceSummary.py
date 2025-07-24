from pydantic import BaseModel

from trainable_entity_extractor.domain.Performance import Performance


class PerformanceSummary(BaseModel):
    training_samples_count: int = 0
    testing_samples_count: int = 0
    methods: list[Performance] = []

    def add_performance(self, method_name: str, performance: float):
        self.methods.append(Performance(method_name=method_name, performance=performance))

    def to_log(self) -> str:
        text = "Performance summary\n"
        text += f"Best method: {self.get_best_method().to_log(self.get_samples_count())}\n"
        text += f"Samples count: {self.get_samples_count()}\n"
        text += f"Train/test split: {self.training_samples_count}/{self.testing_samples_count}\n"
        text += "Methods by performance:\n"
        for method in sorted(self.methods, key=lambda x: x.performance, reverse=True):
            text += f"{method.to_log(self.get_samples_count())}\n"

        return text

    def get_best_method(self) -> Performance:
        if not self.methods:
            return Performance(method_name="No methods", performance=0.0)

        return max(self.methods, key=lambda x: x.performance)

    def get_samples_count(self) -> int:
        return self.training_samples_count + self.testing_samples_count


if __name__ == "__main__":
    performance_summary = PerformanceSummary(training_samples_count=100, testing_samples_count=50)
    performance_summary.add_performance("Method A", 85.5)
    performance_summary.add_performance("Method B", 90.0)
    performance_summary.add_performance("Method C", 78.2)
    performance_summary.add_performance("Method D", 92.3)
    performance_summary.add_performance("Method G", 92.3)
    print(performance_summary.to_log())
