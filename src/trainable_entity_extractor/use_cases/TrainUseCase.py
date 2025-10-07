from pathlib import Path

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.Performance import Performance
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.ports.ExtractorBase import ExtractorBase
from trainable_entity_extractor.ports.Logger import Logger


class TrainUseCase:
    def __init__(self, extractors: list[type[ExtractorBase]], logger: Logger):
        self.extractors: list[type[ExtractorBase]] = extractors
        self.logger = logger

    def train_one_method(
        self, extractor_job: TrainableEntityExtractorJob, extraction_data: ExtractionData
    ) -> tuple[bool, str]:

        method_path = Path(extraction_data.extraction_identifier.get_path()) / extractor_job.method_name
        if method_path.exists() and any(method_path.iterdir()):
            return True, ""

        extractor_name = extractor_job.extractor_name
        for extractor in self.extractors:
            extractor_instance = extractor(extraction_data.extraction_identifier, self.logger)
            if extractor_instance.get_name() != extractor_name:
                continue

            return extractor_instance.train_one_method(extractor_job, extraction_data)

        return False, f"Extractor {extractor_name} not found"

    def get_performance(
        self, extractor_job: TrainableEntityExtractorJob, extraction_data: ExtractionData
    ) -> Performance | None:
        extractor_name = extractor_job.extractor_name
        for extractor in self.extractors:
            extractor_instance = extractor(extraction_data.extraction_identifier, self.logger)
            if extractor_instance.get_name() != extractor_name:
                continue

            return extractor_instance.get_performance(extractor_job, extraction_data)

        return None

    def get_jobs(self, extraction_data: ExtractionData) -> list[TrainableEntityExtractorJob]:
        for extractor in self.extractors:
            extractor_instance = extractor(extraction_data.extraction_identifier, self.logger)

            if not extractor_instance.can_be_used(extraction_data):
                continue

            return extractor_instance.get_distributed_jobs(extraction_data)

        return []
