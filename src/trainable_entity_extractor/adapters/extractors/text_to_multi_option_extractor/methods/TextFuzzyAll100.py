import math

from rapidfuzz import fuzz

from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSamples import PredictionSamples
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import (
    TextToMultiOptionMethod,
)


class TextFuzzyAll100(TextToMultiOptionMethod):

    threshold = 100

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        return True

    def get_appearances(self, text: str, options: list[str]) -> list[str]:
        appearances = []

        for option in options:
            if len(text) < math.ceil(len(option) * 0.85):
                continue

            if fuzz.partial_ratio(option, text.lower()) >= self.threshold:
                appearances.append(option)

        return list(set(appearances))

    def predict_multi_option(self, prediction_samples: PredictionSamples) -> list[list[Option]]:
        predictions: list[list[Option]] = list()
        option_labels = [option.label.lower() for option in prediction_samples.options]
        for sample in prediction_samples.prediction_samples:
            values = self.get_appearances(sample.get_input_text(), option_labels)
            predictions.append([option for option in prediction_samples.options if option.label in values])

        return predictions

    def train(self, multi_option_data: ExtractionData):
        pass
