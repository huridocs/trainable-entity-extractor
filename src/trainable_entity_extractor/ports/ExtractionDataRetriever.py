from abc import ABC, abstractmethod
from typing import Optional
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.PredictionSample import PredictionSample


class ExtractionDataRetriever(ABC):
    @abstractmethod
    def get_extraction_data(self, extraction_identifier: ExtractionIdentifier) -> Optional[ExtractionData]:
        pass

    @abstractmethod
    def cache_extraction_data(self, extraction_identifier: ExtractionIdentifier, extraction_data: ExtractionData) -> bool:
        pass

    @abstractmethod
    def cache_prediction_data(
        self, extraction_identifier: ExtractionIdentifier, prediction_data: list[PredictionSample]
    ) -> bool:
        pass

    @abstractmethod
    def get_prediction_data(self, extraction_identifier: ExtractionIdentifier) -> list[PredictionSample]:
        pass
