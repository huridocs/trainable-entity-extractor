from abc import ABC, abstractmethod

from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.PdfData import PdfData


class SegmentSelectorBase(ABC):
    def __init__(self, extraction_identifier: ExtractionIdentifier):
        self.extraction_identifier = extraction_identifier

    @abstractmethod
    def prepare_model_folder(self):
        pass

    @abstractmethod
    def get_predictions_for_performance(self, training_set: list[PdfData], test_set: list[PdfData]) -> list[int]:
        pass

    def get_name(self):
        return self.__class__.__name__
