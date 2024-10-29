import json
import os
import shutil
from abc import abstractmethod
from copy import deepcopy
from os.path import join, exists
from pathlib import Path

from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanBeginningDot250 import (
    CleanBeginningDot250,
)


class ToTextExtractorMethod:

    def __init__(self, extraction_identifier: ExtractionIdentifier, from_class_name: str = ""):
        self.from_class_name = from_class_name
        self.extraction_identifier = extraction_identifier
        os.makedirs(self.extraction_identifier.get_path(), exist_ok=True)

    def get_path(self):
        if self.from_class_name:
            path = join(self.extraction_identifier.get_path(), self.from_class_name, self.get_name())
        else:
            path = join(self.extraction_identifier.get_path(), self.get_name())

        os.makedirs(path, exist_ok=True)
        return path

    def get_name(self):
        return self.__class__.__name__

    def save_json(self, file_name: str, data: any):
        path = join(self.get_path(), file_name)

        if not exists(Path(path).parent):
            os.makedirs(Path(path).parent)

        with open(path, "w") as file:
            json.dump(data, file)

    def load_json(self, file_name: str):
        path = join(self.get_path(), file_name)

        if not exists(path):
            return ""

        with open(path, "r") as file:
            return json.load(file)

    @abstractmethod
    def train(self, extraction_data: ExtractionData):
        pass

    @abstractmethod
    def predict(self, predictions_samples: list[PredictionSample]) -> list[str]:
        pass

    @staticmethod
    def clean_text(text: str) -> str:
        return " ".join(text.split())

    def performance(self, performance_train_set: ExtractionData, performance_test_set: ExtractionData) -> float:
        if not performance_train_set.samples:
            return 0

        self.train(performance_train_set)
        samples = performance_test_set.samples
        predictions = self.predict(
            [
                PredictionSample(pdf_data=deepcopy(x.pdf_data), segment_selector_texts=x.segment_selector_texts)
                for x in samples
            ]
        )

        correct = [
            sample
            for sample, prediction in zip(performance_test_set.samples, predictions)
            if self.clean_text(sample.labeled_data.label_text) == self.clean_text(prediction)
        ]
        return 100 * len(correct) / len(performance_test_set.samples)

    def log_performance_sample(self, extraction_data: ExtractionData, predictions: list[str]):
        config_logger.info(f"Performance predictions for {self.get_name()}")
        filtered_extraction_data = CleanBeginningDot250().filter(extraction_data)
        message = ""
        for i, (training_sample, prediction) in enumerate(zip(extraction_data.samples, predictions)):
            if i >= 5:
                break
            segments = filtered_extraction_data.samples[i].pdf_data.pdf_data_segments
            document_text = " ".join([x.text_content for x in segments])
            message += f"\nprediction            : {prediction[:70].strip()}\n"
            message += f"truth                 : {training_sample.labeled_data.label_text[:70].strip()}\n"
            message += f"segment selector text : {' '.join(training_sample.segment_selector_texts)[:70].strip()}\n"
            message += f"document text         : {document_text[:70].strip()}\n"

        config_logger.info(message)

    def remove_method_data(self):
        shutil.rmtree(self.get_path(), ignore_errors=True)
