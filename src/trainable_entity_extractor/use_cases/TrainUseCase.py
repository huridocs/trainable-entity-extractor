from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.Performance import Performance
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.ports.ExtractorBase import ExtractorBase


class TrainUseCase:
    def __init__(self, extractors: list[type[ExtractorBase]]):
        self.extractors: list[type[ExtractorBase]] = extractors

    def train_one_method(
        self, extractor_job: TrainableEntityExtractorJob, extraction_data: ExtractionData
    ) -> tuple[bool, str]:
        extractor_name = extractor_job.extractor_name
        for extractor in self.extractors:
            extractor_instance = extractor(extraction_data.extraction_identifier)
            if extractor_instance.get_name() != extractor_name:
                continue

            return extractor_instance.train_one_method(extractor_job, extraction_data)

        return False, f"Extractor {extractor_name} not found"

    def get_performance(
        self, extractor_job: TrainableEntityExtractorJob, extraction_data: ExtractionData
    ) -> Performance | None:
        extractor_name = extractor_job.extractor_name
        for extractor in self.extractors:
            extractor_instance = extractor(extraction_data.extraction_identifier)
            if extractor_instance.get_name() != extractor_name:
                continue

            return extractor_instance.get_performance(extractor_job, extraction_data)

        return None

    def get_jobs(self, extraction_data: ExtractionData) -> list[TrainableEntityExtractorJob]:
        jobs = list()
        for extractor in self.extractors:
            extractor_instance = extractor(extraction_data.extraction_identifier)

            if not extractor_instance.can_be_used(extraction_data):
                continue

            jobs = extractor_instance.get_distributed_jobs(extraction_data)

        return jobs
