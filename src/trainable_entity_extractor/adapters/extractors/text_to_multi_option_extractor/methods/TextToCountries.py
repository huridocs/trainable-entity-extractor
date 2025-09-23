import json
import re
import unicodedata

from pydantic import BaseModel
from country_named_entity_recognition import find_countries

from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import (
    TextToMultiOptionMethod,
)


class OptionKeyword(BaseModel):
    keyword: str
    option: Option
    is_country_name: bool = False


class TextToCountries(TextToMultiOptionMethod):

    OPTION_LABELS_FILE_NAME = "option_labels.json"

    @staticmethod
    def _find_countries(text: str) -> list:
        return [x[0] for x in find_countries(text, is_ignore_case=True)]

    @staticmethod
    def clean_text(text: str) -> str:
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text.lower()

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        total_options = len(extraction_data.options)
        option_labels = [option.label for option in extraction_data.options]

        percentage_matched = sum(1 for label in option_labels if self._find_countries(label)) / total_options
        return percentage_matched > 0.5

    def predict(self, prediction_samples_data: PredictionSamplesData) -> list[list[Option]]:
        predictions: list[list[Option]] = list()
        for sample in prediction_samples_data.prediction_samples:
            country_codes = self.get_country_codes(sample.get_input_text())
            predictions.append([option for option in prediction_samples_data.options if option.label in country_codes])

        return predictions

    def _load_option_keywords(self) -> list[OptionKeyword]:
        try:
            json_data = self.load_json(self.OPTION_LABELS_FILE_NAME)
            option_keywords = [OptionKeyword(**json.loads(item)) for item in json_data]
            return option_keywords
        except Exception:
            return []

    def train(self, multi_option_data: ExtractionData):
        options_keywords: list[OptionKeyword] = list()
        for option in self.options:
            label = option.label

            detected_countries = find_countries(label)
            if detected_countries:
                country_name = detected_countries[0][0]
                options_keywords.append(OptionKeyword(keyword=country_name.name, option=option, is_country_name=True))
            else:
                options_keywords.append(OptionKeyword(keyword=self.clean_text(label), option=option, is_country_name=False))

        self.save_json(self.OPTION_LABELS_FILE_NAME, [x.model_dump_json() for x in options_keywords])

    def get_no_country_strings(self, text: str, option_keywords: list[OptionKeyword]) -> list[Option]:
        found_options = []
        text_clean = self.clean_text(text)
        sorted_keywords = sorted(option_keywords, key=lambda x: len(x.keyword), reverse=True)
        for option_keyword in sorted_keywords:
            if option_keyword.is_country_name:
                continue
            keyword_clean = self.clean_text(option_keyword.keyword)
            if keyword_clean and keyword_clean in text_clean:
                found_options.append(option_keyword.option)
                text_clean = text_clean.replace(keyword_clean, "", 1)
        return found_options
