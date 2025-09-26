from abc import ABC, abstractmethod
from typing import Optional
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.Suggestion import Suggestion


class ExtractionDataRetriever(ABC):
    @abstractmethod
    def get_extraction_data(self, extraction_identifier: ExtractionIdentifier) -> Optional[ExtractionData]:
        pass

    @abstractmethod
    def save_extraction_data(self, extraction_identifier: ExtractionIdentifier, extraction_data: ExtractionData) -> bool:
        pass

    @abstractmethod
    def save_prediction_data(
        self, extraction_identifier: ExtractionIdentifier, prediction_data: list[PredictionSample]
    ) -> bool:
        pass

    @abstractmethod
    def get_prediction_data(self, extraction_identifier: ExtractionIdentifier) -> list[PredictionSample]:
        pass

    @abstractmethod
    def get_suggestions(self, extraction_identifier: ExtractionIdentifier) -> list[Suggestion]:
        pass

    @abstractmethod
    def save_suggestions(self, extraction_identifier: ExtractionIdentifier, suggestions: list[Suggestion]) -> bool:
        pass

    @abstractmethod
    def is_extractor_cancelled(self, extractor_identifier: ExtractionIdentifier):
        pass
