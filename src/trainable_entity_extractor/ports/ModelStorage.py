from abc import ABC, abstractmethod
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob


class ModelStorage(ABC):
    """Abstract interface for model storage and retrieval operations"""

    @abstractmethod
    def upload_model(
        self, extraction_identifier: ExtractionIdentifier, method_name: str, extractor_job: TrainableEntityExtractorJob
    ) -> bool:
        """Upload a trained model to storage"""
        pass

    @abstractmethod
    def download_model(self, extraction_identifier: ExtractionIdentifier) -> bool:
        """Download a model from storage"""
        pass

    @abstractmethod
    def check_model_completion_signal(self, extraction_identifier: ExtractionIdentifier) -> bool:
        """Check if model upload is completed by checking for completion signal file"""
        pass

    @abstractmethod
    def create_model_completion_signal(self, extraction_identifier: ExtractionIdentifier) -> bool:
        """Create a completion signal file after successful model upload"""
        pass
