from abc import ABC, abstractmethod

from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.ports.MethodBase import MethodBase


class SegmentSelectorBase(MethodBase):

    @abstractmethod
    def prepare_model_folder(self):
        pass

    @abstractmethod
    def get_predictions_for_performance(self, training_set: list[PdfData], test_set: list[PdfData]) -> list[int]:
        pass

    def get_name(self):
        return self.__class__.__name__
