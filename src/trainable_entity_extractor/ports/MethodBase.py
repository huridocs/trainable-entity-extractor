import os
from abc import ABC, abstractmethod
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PredictionSamples import PredictionSamples


class MethodBase(ABC):
    def __init__(self, extraction_identifier: ExtractionIdentifier, from_class_name: str = ""):
        self.extraction_identifier = extraction_identifier
        self.from_class_name = from_class_name
        os.makedirs(self.extraction_identifier.get_path(), exist_ok=True)

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
    def predict(self, prediction_samples: PredictionSamples) -> list[str] | list[list[Option]]:
        pass

    def should_be_retrained_with_more_data(self) -> bool:
        return True

    def remove_method_data(self) -> None:
        pass
