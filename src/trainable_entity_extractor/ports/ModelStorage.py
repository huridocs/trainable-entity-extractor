from abc import ABC, abstractmethod
from typing import Optional
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
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
