from abc import ABC, abstractmethod
from typing import Optional
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob


class ModelStorage(ABC):

    @abstractmethod
    def upload_model(self, extraction_identifier: ExtractionIdentifier, extractor_job: TrainableEntityExtractorJob) -> bool:
        pass

    @abstractmethod
    def download_model(self, extraction_identifier: ExtractionIdentifier) -> bool:
        pass

    @abstractmethod
    def check_model_completion_signal(self, extraction_identifier: ExtractionIdentifier) -> bool:
        pass

    @abstractmethod
    def create_model_completion_signal(self, extraction_identifier: ExtractionIdentifier) -> bool:
        pass

    @abstractmethod
    def get_extractor_job(self, extraction_identifier: ExtractionIdentifier) -> Optional[TrainableEntityExtractorJob]:
        pass

    @staticmethod
    def serialize_job_to_dict(job: TrainableEntityExtractorJob) -> dict:
        return {
            "version": "1.0",  # Version for future compatibility
            "run_name": job.run_name,
            "extraction_name": job.extraction_name,
            "extractor_name": job.extractor_name,
            "method_name": job.method_name,
            "multi_value": job.multi_value,
            "options": [option.model_dump() for option in job.options],
            "gpu_needed": job.gpu_needed,
            "timeout": job.timeout,
            "metadata": {},
        }

    @staticmethod
    def deserialize_job_from_dict(job_data: dict) -> TrainableEntityExtractorJob:
        version = job_data.get("version", "1.0")

        run_name = job_data.get("run_name", "")
        extraction_name = job_data.get("extraction_name", "")
        extractor_name = job_data.get("extractor_name", "")
        method_name = job_data.get("method_name", "")
        multi_value = job_data.get("multi_value", False)
        options_data = job_data.get("options", [])
        options = [Option(**option_data) for option_data in options_data]
        gpu_needed = job_data.get("gpu_needed", False)
        timeout = job_data.get("timeout", 3600)

        additional_fields = {}
        if version != "1.0":
            pass

        return TrainableEntityExtractorJob(
            run_name=run_name,
            extraction_name=extraction_name,
            extractor_name=extractor_name,
            method_name=method_name,
            multi_value=multi_value,
            options=options,
            gpu_needed=gpu_needed,
            timeout=timeout,
        )
