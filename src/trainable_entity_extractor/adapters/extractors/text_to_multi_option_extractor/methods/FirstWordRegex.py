import re

from pydantic import BaseModel
from tdda import *
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import (
    TextToMultiOptionMethod,
)


class OptionRegex(BaseModel):
    option_id: str
    regex_list: list[str]


class FirstWordRegex(TextToMultiOptionMethod):
    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        return True

    def predict(self, prediction_samples: PredictionSamplesData) -> list[list[Option]]:
        predictions: list[list[Option]] = list()
        options_regex: list[OptionRegex] = [self.load_option_regex(option) for option in prediction_samples.options]
        options_regex.sort(key=lambda x: len(x.regex_list))
        for sample in prediction_samples.prediction_samples:
            options_ids = self.predict_one(sample.get_input_text(), options_regex)
            if options_ids:
                predictions.append([option for option in prediction_samples.options if option.id in options_ids])
            else:
                predictions.append(
                    [option for option in prediction_samples.options if option.id == options_regex[-1].option_id]
                )

        return predictions

    @staticmethod
    def get_option_regex_file_name(option: Option) -> str:
        option_name = option.id.strip().replace(" ", "_")
        return f"{option_name}_regex_list.json"

    def train(self, multi_option_data: ExtractionData):
        for option in multi_option_data.options:
            texts = [sample.get_input_text() for sample in multi_option_data.samples if option in sample.labeled_data.values]
            first_words = [text.split()[0] for text in texts if text]
            regex_list = rexpy.extract(first_words)
            regex_list = [regex[1:-1] for regex in regex_list]
            self.save_json(self.get_option_regex_file_name(option), regex_list)

    def load_option_regex(self, option: Option) -> OptionRegex:
        if not self.get_path(self.get_option_regex_file_name(option)).exists():
            return OptionRegex(option_id=option.id, regex_list=[])

        regex_list = self.load_json(self.get_option_regex_file_name(option))
        return OptionRegex(option_id=option.id, regex_list=regex_list)

    def predict_one(self, source_text: str, options_regex: list[OptionRegex]) -> list[str]:
        if not source_text.strip():
            return []

        first_word = source_text.split()[0]
        predictions = []
        if not first_word:
            return []

        for option_regex in options_regex:
            for regex in option_regex.regex_list:
                match = re.match(regex, first_word)
                if match:
                    predictions.append(option_regex.option_id)
                    break

            if not self.multi_value and predictions:
                break

        return predictions
