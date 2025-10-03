import os
import json
from typing import Optional

from trainable_entity_extractor.config import EXTRACTOR_JOB_PATH
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.ports.ModelStorage import ModelStorage


class LocalModelStorage(ModelStorage):

    def upload_model(self, extraction_identifier: ExtractionIdentifier, extractor_job: TrainableEntityExtractorJob) -> bool:
        return self.save_extractor_job(extraction_identifier, extractor_job)

    def download_model(self, extraction_identifier: ExtractionIdentifier) -> bool:
        try:
            model_path = extraction_identifier.get_path()
            return os.path.exists(model_path)
        except Exception:
            return False

    def get_extractor_job(self, extraction_identifier: ExtractionIdentifier) -> Optional[TrainableEntityExtractorJob]:
        try:
            model_path = extraction_identifier.get_path()
            job_file_path = os.path.join(model_path, EXTRACTOR_JOB_PATH)

            if os.path.exists(job_file_path):
                with open(job_file_path, "r", encoding="utf-8") as f:
                    job_data = json.load(f)
                return self.deserialize_job_from_dict(job_data)
            return None
        except Exception as e:
            print(f"Error loading job: {e}")
            return None
