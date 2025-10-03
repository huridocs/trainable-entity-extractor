import json
import os
from abc import ABC, abstractmethod
from typing import Optional

from trainable_entity_extractor.config import EXTRACTOR_JOB_PATH
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
    def get_extractor_job(self, extraction_identifier: ExtractionIdentifier) -> Optional[TrainableEntityExtractorJob]:
        pass

    def save_extractor_job(
        self, extraction_identifier: ExtractionIdentifier, extractor_job: TrainableEntityExtractorJob
    ) -> bool:
        try:
            model_path = extraction_identifier.get_path()
            if not os.path.exists(model_path):
                os.makedirs(model_path, exist_ok=True)

            extractor_job_dir = os.path.join(model_path, EXTRACTOR_JOB_PATH.parent)
            if not os.path.exists(extractor_job_dir):
                os.makedirs(extractor_job_dir, exist_ok=True)

            job_file_path = os.path.join(model_path, EXTRACTOR_JOB_PATH)
            job_data = self.serialize_job_to_dict(extractor_job)

            with open(job_file_path, "w", encoding="utf-8") as f:
                json.dump(job_data, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"Error saving job: {e}")
            return False

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
