import json
from abc import abstractmethod
from os import makedirs
from os.path import exists
from pathlib import Path

from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.data.Suggestion import Suggestion
from trainable_entity_extractor.data.TrainingSample import TrainingSample


class ExtractorBase:
    def __init__(self, extraction_identifier: ExtractionIdentifier):
        self.extraction_identifier = extraction_identifier

    def get_name(self):
        return self.__class__.__name__

    @abstractmethod
    def create_model(self, extraction_data: ExtractionData) -> tuple[bool, str]:
        pass

    @abstractmethod
    def get_suggestions(self, predictions_samples: list[PredictionSample]) -> list[Suggestion]:
        pass

    @abstractmethod
    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        pass

    @staticmethod
    def is_multilingual(multi_option_data: ExtractionData) -> bool:
        not_multilingual_languages = ["", "en", "eng"]

        for sample in multi_option_data.samples:
            if sample.labeled_data.language_iso not in not_multilingual_languages:
                return True

        return False

    @staticmethod
    def get_train_test_sets(extraction_data: ExtractionData) -> (ExtractionData, ExtractionData):
        if len(extraction_data.samples) < 8:
            return extraction_data, extraction_data

        train_size = int(len(extraction_data.samples) * 0.8)

        train_set: list[TrainingSample] = extraction_data.samples[:train_size]

        if len(extraction_data.samples) < 15:
            test_set: list[TrainingSample] = extraction_data.samples[-10:]
        else:
            test_set = extraction_data.samples[train_size:]

        train_extraction_data = ExtractorBase.get_extraction_data_from_samples(extraction_data, train_set)
        test_extraction_data = ExtractorBase.get_extraction_data_from_samples(extraction_data, test_set)
        return train_extraction_data, test_extraction_data

    @staticmethod
    def get_extraction_data_from_samples(extraction_data: ExtractionData, samples: list[TrainingSample]) -> ExtractionData:
        return ExtractionData(
            samples=samples,
            options=extraction_data.options,
            multi_value=extraction_data.multi_value,
            extraction_identifier=extraction_data.extraction_identifier,
        )

    @staticmethod
    def save_json(path: str, data: any):
        if not exists(Path(path).parent):
            makedirs(Path(path).parent)

        with open(path, "w") as file:
            json.dump(data, file)
