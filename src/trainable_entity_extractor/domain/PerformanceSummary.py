from pydantic import BaseModel, Field
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.Performance import Performance
from time import time

from trainable_entity_extractor.use_cases.send_logs import send_logs


class PerformanceSummary(BaseModel):
    extractor_name: str = "Unknown Extractor"
    samples_count: int = 0
    options_count: int = 0
    languages: list[str] = list()
    training_samples_count: int = 0
    testing_samples_count: int = 0
    performances: list[Performance] = []
    extraction_identifier: ExtractionIdentifier | None = None
    previous_timestamp: int = Field(default_factory=lambda: int(time()))
    empty_pdf_count: int = 0

    def add_performance(self, method_name: str, performance: float):
        current_time = int(time())
        performance = Performance(
            method_name=method_name, performance=performance, execution_seconds=int(current_time - self.previous_timestamp)
        )
        self.previous_timestamp = current_time
        self.performances.append(performance)
        send_logs(self.extraction_identifier, f"Performance {performance.to_log(self.testing_samples_count)}")

    def to_log(self) -> str:
        total_time = sum(performance.execution_seconds for performance in self.performances)

        text = "Performance summary\n"
        text += f"Id: {self.extraction_identifier} / {self.extractor_name}\n" if self.extraction_identifier else ""
        text += f"Best method: {self.get_best_method().to_log(self.testing_samples_count)}\n"
        text += f"Training time: {Performance.get_execution_time_string(total_time)}\n"
        text += f"Samples: {self.samples_count}\n"
        text += f"Train/test: {self.training_samples_count}/{self.testing_samples_count}\n"
        text += f"{len(self.languages)} language(s): {', '.join(self.languages) if self.languages else 'None'}\n"
        text += f"Empty PDFs: {self.empty_pdf_count}\n" if self.empty_pdf_count else ""
        text += f"Options count: {self.options_count}\n" if self.options_count > 0 else ""
        text += "Methods by performance:\n"
        for performance in sorted(self.performances, key=lambda x: x.performance, reverse=True):
            text += f"{performance.to_log(self.testing_samples_count)}\n"

        return text

    def get_best_method(self) -> Performance:
        if not self.performances:
            return Performance(method_name="No methods", performance=0.0)

        return max(self.performances, key=lambda x: x.performance)

    @staticmethod
    def from_extraction_data(
        extractor_name: str, training_samples_count: int, testing_samples_count: int, extraction_data: ExtractionData
    ) -> "PerformanceSummary":
        languages = set()
        empty_pdf_count = 0
        for sample in extraction_data.samples:
            if sample.labeled_data and sample.labeled_data.language_iso:
                languages.add(sample.labeled_data.language_iso)

            if sample.pdf_data and sample.pdf_data.pdf_features and not sample.pdf_data.get_text():
                empty_pdf_count += 1

        return PerformanceSummary(
            extraction_identifier=extraction_data.extraction_identifier,
            extractor_name=extractor_name,
            samples_count=len(extraction_data.samples),
            options_count=len(extraction_data.options) if extraction_data.options else 0,
            languages=list(languages),
            training_samples_count=training_samples_count,
            testing_samples_count=testing_samples_count,
            empty_pdf_count=empty_pdf_count,
        )
