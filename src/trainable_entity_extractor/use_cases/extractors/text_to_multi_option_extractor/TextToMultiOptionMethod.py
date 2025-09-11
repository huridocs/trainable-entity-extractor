import json
import os
import shutil
from abc import abstractmethod
from os.path import join, exists
from pathlib import Path

from numpy import argmax
from sklearn.metrics import f1_score
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.use_cases.extractors.MethodBase import MethodBase


class TextToMultiOptionMethod(MethodBase):
    def __init__(self, extraction_identifier: ExtractionIdentifier, options=None, multi_value: bool = False, method_name=""):
        super().__init__(extraction_identifier)
        if options is None:
            options = []
        self.options = options
        self.multi_value = multi_value
        self.method_name = method_name
        os.makedirs(self.extraction_identifier.get_path(), exist_ok=True)

    def get_name(self):
        return self.__class__.__name__

    def get_path(self, file_name) -> Path:
        return Path(self.extraction_identifier.get_path(), self.get_name(), file_name)

    def save_json(self, file_name: str, data: any):
        path = self.get_path(file_name)
        if not exists(path.parent):
            os.makedirs(path.parent)

        with open(path, "w") as file:
            json.dump(data, file)

    def load_json(self, file_name: str):
        path = self.get_path(file_name)
        with open(path, "r") as file:
            return json.load(file)

    def remove_model(self):
        shutil.rmtree(join(self.extraction_identifier.get_path(), self.get_name()), ignore_errors=True)

    def remove_method_data(self) -> None:
        self.remove_model()

    @abstractmethod
    def predict(self, predictions_samples: list[PredictionSample]) -> list[list[Option]]:
        pass

    def get_performance(self, train_set: ExtractionData, test_set: ExtractionData) -> float:
        """Get performance using standardized train/test sets"""
        if not test_set.samples:
            return 0

        self.train(train_set)

        prediction_samples = [PredictionSample(source_text=x.labeled_data.source_text) for x in test_set.samples]
        predictions = self.predict(prediction_samples)

        self.remove_model()

        correct_one_hot_encoding = self.get_one_hot_encoding(test_set)
        predictions_one_hot_encoding = [
            [1 if option in prediction else 0 for option in self.options] for prediction in predictions
        ]
        return 100 * f1_score(correct_one_hot_encoding, predictions_one_hot_encoding, average="micro")

    def get_one_hot_encoding(self, multi_option_data: ExtractionData):
        options_ids = [option.id for option in self.options]
        one_hot_encoding = list()
        for sample in multi_option_data.samples:
            one_hot_encoding.append([0] * len(options_ids))
            for option in sample.labeled_data.values:
                if option.id not in options_ids:
                    print(f"option {option.id} not in {options_ids}")
                    continue
                one_hot_encoding[-1][options_ids.index(option.id)] = 1

        return one_hot_encoding

    def predictions_to_options_list(self, predictions_scores: list[list[float]]) -> list[list[Option]]:
        return [self.one_prediction_to_option_list(prediction) for prediction in predictions_scores]

    def one_prediction_to_option_list(self, prediction_scores: list[float]) -> list[Option]:
        if not self.multi_value:
            best_score_index = argmax(prediction_scores)
            return [self.options[best_score_index]] if prediction_scores[best_score_index] > 0.5 else []

        return [self.options[i] for i, value in enumerate(prediction_scores) if value > 0.5]

    @staticmethod
    def get_text(text: str):
        words = list()

        for word in text.split():
            clean_word = "".join([x for x in word if x.isalpha() or x.isdigit()])

            if clean_word:
                words.append(clean_word)

        return " ".join(words)

    @abstractmethod
    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        pass
