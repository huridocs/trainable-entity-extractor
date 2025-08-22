import json
import os
import shutil
from abc import ABC, abstractmethod
from os import makedirs
from os.path import exists, join
from pathlib import Path

from numpy import argmax

from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.Value import Value


class MultiLabelMethod(ABC):
    def __init__(
        self, extraction_identifier: ExtractionIdentifier, options: list[Option], multi_value: bool, method_name: str = ""
    ):
        self.extraction_identifier = extraction_identifier
        self.options = options
        self.multi_value = multi_value
        self.method_name = method_name

    def get_name(self):
        return self.__class__.__name__

    def get_path(self):
        if self.method_name:
            path = join(self.extraction_identifier.get_path(), self.method_name)
        else:
            path = join(self.extraction_identifier.get_path(), self.get_name())

        os.makedirs(path, exist_ok=True)
        return path

    @abstractmethod
    def train(self, multi_option_data: ExtractionData):
        pass

    @abstractmethod
    def predict(self, multi_option_data: ExtractionData) -> list[list[Value]]:
        pass

    def save_json(self, file_name: str, data: any):
        path = join(self.get_path(), file_name)
        if not exists(Path(path).parent):
            makedirs(Path(path).parent)

        with open(path, "w") as file:
            json.dump(data, file)

    def load_json(self, file_name: str):
        path = join(self.get_path(), file_name)

        with open(path, "r") as file:
            return json.load(file)

    def remove_model(self):
        shutil.rmtree(join(self.get_path()), ignore_errors=True)

    def get_texts_labels(self, multi_option_data: ExtractionData) -> (list[str], list[list[int]]):
        texts = list()
        for sample in multi_option_data.samples:
            texts.append(" ".join([x.text_content.strip() for x in sample.pdf_data.pdf_data_segments]))

        labels = self.get_one_hot_encoding(multi_option_data)
        return texts, labels

    def predictions_to_options_list(self, predictions) -> list[list[Value]]:
        return [self.one_prediction_to_option_list(prediction) for prediction in predictions]

    def one_prediction_to_option_list(self, prediction) -> list[Value]:
        if not self.multi_value:
            best_score_index = argmax(prediction)
            return [self.options[best_score_index]] if prediction[best_score_index] > 0.5 else []

        return [Value.from_option(self.options[i]) for i, value in enumerate(prediction) if value > 0.5]

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

    @staticmethod
    def can_be_used(extraction_data: ExtractionData) -> bool:
        pass
