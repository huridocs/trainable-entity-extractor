from pydantic import BaseModel

from trainable_entity_extractor.domain.Performance import Performance


class PerformanceSummary(BaseModel):
    samples_count: int
    methods: list[Performance] = list()

    def add_performance(self, method_name: str, performance: float):
        self.methods.append(Performance(method_name=method_name, performance=performance))

    def to_log(self) -> str:
        text = "Performance summary\n"
        text += "-------------------\n"
        text += f"Samples count: {self.samples_count}\n"
        text += f"Best method: {self.get_best_method()}\n"
        text += "Methods by performance:\n"
        for method in sorted(self.methods, key=lambda x: x.performance, reverse=True):
            text += f"{method}\n"

        return text

    def get_best_method(self) -> Performance:
        if not self.methods:
            return Performance(method_name="No methods", performance=0.0)

        return max(self.methods, key=lambda x: x.performance)


if __name__ == "__main__":
    performance_summary = PerformanceSummary(samples_count=100)
    performance_summary.add_performance("Method A", 85.5)
    performance_summary.add_performance("Method B", 90.0)
    performance_summary.add_performance("Method C", 78.2)
    performance_summary.add_performance("Method D", 92.3)
    performance_summary.add_performance("Method G", 92.3)
    print(performance_summary.to_log())
