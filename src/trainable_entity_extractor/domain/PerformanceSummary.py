from pydantic import BaseModel, Field

from trainable_entity_extractor.domain.DistributedJob import DistributedJob
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.PerformanceLog import PerformanceLog
from time import time


class PerformanceSummary(BaseModel):
    extractor_name: str = "Unknown Extractor"
    samples_count: int = 0
    options_count: int = 0
    languages: list[str] = list()
    training_samples_count: int = 0
    testing_samples_count: int = 0
    performances: list[PerformanceLog] = []
    extraction_identifier: ExtractionIdentifier | None = None
    previous_timestamp: int = Field(default_factory=lambda: int(time()))
    empty_pdf_count: int = 0

    def add_performance(self, method_name: str, performance: float, failed: bool = False):
        current_time = int(time())
        performance = PerformanceLog(
            method_name=method_name,
            performance=performance,
            execution_seconds=int(current_time - self.previous_timestamp),
            failed=failed,
        )
        self.previous_timestamp = current_time
        self.performances.append(performance)

    def add_performance_from_sub_job(self, sub_job):
        """Add performance data from a DistributedSubJob"""
        if sub_job.result and hasattr(sub_job.result, "performance_score"):
            performance_score = sub_job.result.performance_score
        elif sub_job.result and hasattr(sub_job.result, "performance"):
            performance_score = sub_job.result.performance
        else:
            performance_score = 0.0

        failed = sub_job.result is None or (hasattr(sub_job.result, "failed") and sub_job.result.failed)

        self.add_performance(sub_job.extractor_job.method_name, performance_score, failed)

    def to_log(self) -> str:
        total_time = sum(performance.execution_seconds for performance in self.performances)

        text = "Performance summary\n"
        text += f"Id: {self.extraction_identifier} / {self.extractor_name}\n" if self.extraction_identifier else ""
        text += f"Best method: {self.get_best_method().to_log(self.testing_samples_count)}\n"
        text += f"Training time: {PerformanceLog.get_execution_time_string(total_time)}\n"
        text += f"Samples: {self.samples_count}\n"
        text += f"Train/test: {self.training_samples_count}/{self.testing_samples_count}\n"
        text += f"{len(self.languages)} language(s): {', '.join(self.languages) if self.languages else 'None'}\n"
        text += f"Empty PDFs: {self.empty_pdf_count}\n" if self.empty_pdf_count else ""
        text += f"Options count: {self.options_count}\n" if self.options_count > 0 else ""
        text += "Methods by performance:\n"
        for performance in sorted(self.performances, key=lambda x: x.performance, reverse=True):
            text += f"{performance.to_log(self.testing_samples_count)}\n"

        return text

    def get_best_method(self) -> PerformanceLog:
        if not self.performances:
            return PerformanceLog(method_name="No methods", performance=0.0)

        return max(self.performances, key=lambda x: x.performance)

    @staticmethod
    def from_distributed_job(distributed_job: DistributedJob) -> "PerformanceSummary":
        testing_samples_count = 0
        training_samples_count = 0
        samples_count = 0
        options_count = 0
        extractor_name = "Unknown Extractor"
        languages = []

        for sub_job in distributed_job.sub_jobs:
            if not sub_job.result:
                continue
            testing_samples_count = sub_job.result.testing_samples_count
            training_samples_count = sub_job.result.training_samples_count
            samples_count = sub_job.result.samples_count
            options_count = len(sub_job.extractor_job.options) if sub_job.extractor_job.options else 0
            extractor_name = sub_job.extractor_job.extractor_name
            languages = sub_job.extractor_job.languages if sub_job.extractor_job.languages else []

        return PerformanceSummary(
            extraction_identifier=distributed_job.extraction_identifier,
            extractor_name=extractor_name,
            samples_count=samples_count,
            options_count=options_count,
            languages=languages,
            training_samples_count=training_samples_count,
            testing_samples_count=testing_samples_count,
            empty_pdf_count=0,
        )
