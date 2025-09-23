import json
import os
import shutil
from os.path import join, exists
from pathlib import Path

from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanBeginningDot250 import (
    CleanBeginningDot250,
)
from trainable_entity_extractor.ports.MethodBase import MethodBase


class ToTextExtractorMethod(MethodBase):
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

    def remove_method_data(self):
        shutil.rmtree(self.get_path(), ignore_errors=True)

    @staticmethod
    def clean_text(text: str) -> str:
        return " ".join(text.split())

    def get_performance(self, train_set: ExtractionData, test_set: ExtractionData) -> float:
        """Get performance using standardized train/test sets"""
        if not train_set.samples:
            return 0

        self.train(train_set)
        samples = test_set.samples
        prediction_samples = PredictionSamples(
            samples=[
                PredictionSample(
                    pdf_data=x.pdf_data.model_copy() if x.pdf_data else None, segment_selector_texts=x.segment_selector_texts
                )
                for x in samples
            ])

        predictions = self.predict(prediction_samples)

        correct = [
            sample
            for sample, prediction in zip(test_set.samples, predictions)
            if self.clean_text(sample.labeled_data.label_text) == self.clean_text(prediction)
        ]
        return 100 * len(correct) / len(test_set.samples)

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
