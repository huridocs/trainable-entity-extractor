import os
from abc import ABC, abstractmethod
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.Value import Value


class MethodBase(ABC):
    def __init__(self, extraction_identifier: ExtractionIdentifier, from_class_name: str = ""):
        self.extraction_identifier = extraction_identifier
        self.from_class_name = from_class_name
        os.makedirs(self.extraction_identifier.get_path(), exist_ok=True)

    @abstractmethod
    def get_name(self) -> str:
        pass

    def get_samples_for_context(self, prediction_samples_data: PredictionSamplesData) -> list[PredictionSample]:
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
    def predict(self, prediction_samples_data: PredictionSamplesData) -> list[str] | list[list[Value]]:
        pass

    def should_be_retrained_with_more_data(self) -> bool:
        return True

    def remove_method_data(self) -> None:
        pass
