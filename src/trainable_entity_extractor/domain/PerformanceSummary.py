from pydantic import BaseModel

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.Performance import Performance


class PerformanceSummary(BaseModel):
    extractor_name: str = "Unknown Extractor"
    samples_count: int = 0
    options_count: int = 0
    languages: list[str] = list()
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
        text += f"{len(self.languages)} language(s): {', '.join(self.languages) if self.languages else 'None'}\n"
        text += f"Options count: {self.options_count}\n" if self.options_count > 0 else ""
        text += "Methods by performance:\n"
        for method in sorted(self.methods, key=lambda x: x.performance, reverse=True):
            text += f"{method.to_log(self.testing_samples_count)}\n"

        return text

    def get_best_method(self) -> Performance:
        if not self.methods:
            return Performance(method_name="No methods", performance=0.0)

        return max(self.methods, key=lambda x: x.performance)

    @staticmethod
    def from_extraction_data(
        extractor_name: str, training_samples_count: int, testing_samples_count: int, extraction_data: ExtractionData
    ) -> "PerformanceSummary":
        languages = set()
        for sample in extraction_data.samples:
            if sample.labeled_data and sample.labeled_data.language_iso:
                languages.add(sample.labeled_data.language_iso)

        return PerformanceSummary(
            extractor_name=extractor_name,
            samples_count=len(extraction_data.samples),
            options_count=len(extraction_data.options) if extraction_data.options else 0,
            languages=list(languages),
            training_samples_count=training_samples_count,
            testing_samples_count=testing_samples_count,
        )
