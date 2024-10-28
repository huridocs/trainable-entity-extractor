import math

from rapidfuzz import fuzz

from trainable_entity_extractor.data.Option import Option
from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import (
    TextToMultiOptionMethod,
)


class TextFuzzyLast(TextToMultiOptionMethod):

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        return True

    @staticmethod
    def get_appearance(texts: list[str], options: list[str]) -> list[str]:
        all_text = " ".join(texts).lower()
        max_words = max([len(option.split()) for option in options])
        words = all_text.split()
        window_texts = [" ".join(words[i : i + max_words]) for i in range(len(words) - max_words + 1)]
        for text in reversed(window_texts):
            for ratio_threshold in range(100, 69, -10):
                for option in options:
                    if len(text) < math.ceil(len(option) * ratio_threshold / 100):
                        continue

                    if fuzz.partial_ratio(option, text.lower()) >= ratio_threshold:
                        return [option]

        return []

    def predict(self, predictions_samples: list[PredictionSample]) -> list[list[Option]]:
        predictions: list[list[Option]] = list()
        option_labels = [option.label.lower() for option in self.options]
        for sample in predictions_samples:
            values = self.get_appearance(sample.tags_texts, option_labels)
            predictions.append([option for option in self.options if option.label in values])

        return predictions

    def train(self, multi_option_data: ExtractionData):
        pass
