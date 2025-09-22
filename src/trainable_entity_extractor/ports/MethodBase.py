from abc import ABC, abstractmethod
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.PredictionSample import PredictionSample


class MethodBase(ABC):
    def __init__(self, extraction_identifier: ExtractionIdentifier):
        self.extraction_identifier = extraction_identifier

    @abstractmethod
    def get_name(self) -> str:
        pass

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        return True

    @abstractmethod
    def get_performance(self, train_set: ExtractionData, test_set: ExtractionData) -> float:
        pass

    @abstractmethod
    def train(self, extraction_data: ExtractionData) -> None:
        pass

    @abstractmethod
    def predict(self, predictions_samples: list[PredictionSample]) -> list[str]:
        pass

    def should_be_retrained_with_more_data(self) -> bool:
        return True

    def remove_method_data(self) -> None:
        pass
