from pydantic import BaseModel

from trainable_entity_extractor.domain.Performance import Performance


class PerformanceSummary(BaseModel):
    extractor_name: str = "Unknown Extractor"
    samples_count: int = 0
    training_samples_count: int = 0
    testing_samples_count: int = 0
    methods: list[Performance] = []

    def add_performance(self, method_name: str, performance: float):
        self.methods.append(Performance(method_name=method_name, performance=performance))

    def to_log(self) -> str:
        text = "Performance summary\n"
        text += f"Extractor: {self.extractor_name}\n"
        text += f"Best method: {self.get_best_method().to_log(self.testing_samples_count)}\n"
        text += f"Samples: {self.samples_count}\n"
        text += f"Train/test: {self.training_samples_count}/{self.testing_samples_count}\n"
        text += "Methods by performance:\n"
        for method in sorted(self.methods, key=lambda x: x.performance, reverse=True):
            text += f"{method.to_log(self.testing_samples_count)}\n"

        return text

    def get_best_method(self) -> Performance:
        if not self.methods:
            return Performance(method_name="No methods", performance=0.0)

        return max(self.methods, key=lambda x: x.performance)


if __name__ == "__main__":
    performance_summary = PerformanceSummary(
        extractor_name="Extractor Example", samples_count=150, training_samples_count=100, testing_samples_count=50
    )
    performance_summary.add_performance("Method A", 98)
    performance_summary.add_performance("Method B", 90.0)
    performance_summary.add_performance("Method C", 78.2)
    performance_summary.add_performance("Method D", 92.3)
    performance_summary.add_performance("Method G", 92.3)
    print(performance_summary.to_log())
