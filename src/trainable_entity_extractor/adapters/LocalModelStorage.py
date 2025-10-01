import os
import json
from typing import Optional

from trainable_entity_extractor.config import EXTRACTOR_JOB_PATH, CACHE_PATH
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.ports.ModelStorage import ModelStorage


class LocalModelStorage(ModelStorage):
    def __init__(self):
        self.completion_signals = {}

    def upload_model(self, extraction_identifier: ExtractionIdentifier, extractor_job: TrainableEntityExtractorJob) -> bool:
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

    def download_model(self, extraction_identifier: ExtractionIdentifier) -> bool:
        """Download/load model locally"""
        try:
            model_path = extraction_identifier.get_path()
            return os.path.exists(model_path)
        except Exception:
            return False

    def check_model_completion_signal(self, extraction_identifier: ExtractionIdentifier) -> bool:
        key = f"{extraction_identifier.run_name}_{extraction_identifier.extraction_name}"
        return self.completion_signals.get(key, False)

    def create_model_completion_signal(self, extraction_identifier: ExtractionIdentifier) -> bool:
        try:
            key = f"{extraction_identifier.run_name}_{extraction_identifier.extraction_name}"
            self.completion_signals[key] = True

            # Also create a physical completion signal file
            completion_file = os.path.join(extraction_identifier.get_path(), "training_complete.signal")
            with open(completion_file, "w") as f:
                f.write("Training completed successfully")

            return True
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
