import math
import unicodedata
from collections import Counter

from rapidfuzz import fuzz

from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSamples import PredictionSamples
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import (
    TextToMultiOptionMethod,
)


class TextFuzzyFirstCleanLabels(TextToMultiOptionMethod):

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        return True

    def get_appearance(self, text: str, options: list[str]) -> list[str]:
        all_text = self.remove_accents(text).lower()
        max_words = max([len(option.split()) for option in options])
        words = all_text.split()
        window_texts = [" ".join(words[i : i + max_words]) for i in range(len(words) - max_words + 1)]
        for text in window_texts:
            for ratio_threshold in range(100, 69, -10):
                for option in options:
                    if len(text) < math.ceil(len(option) * ratio_threshold / 100):
                        continue

                    if fuzz.partial_ratio(option, text.lower()) >= ratio_threshold:
                        return [option]

        return []

    def predict_multi_option(self, prediction_samples: PredictionSamples) -> list[list[Option]]:
        predictions: list[list[Option]] = list()
        option_labels = self.get_cleaned_labels(prediction_samples.options)
        for sample in prediction_samples.prediction_samples:
            values = self.get_appearance(sample.get_input_text(), option_labels)
            predictions.append([option for option in prediction_samples.options if self.remove_accents(option.label).lower() in values])

        return predictions

    def train(self, multi_option_data: ExtractionData):
        pass

    @staticmethod
    def remove_accents(text: str):
        nfkd_form = unicodedata.normalize("NFKD", text)
        only_ascii = nfkd_form.encode("ASCII", "ignore")
        return only_ascii.decode()

    def get_cleaned_labels(self, options: list[Option]) -> list[str]:
        options_labels = [self.remove_accents(x.label.lower()) for x in options]
        words_counter = Counter()
        for option in options_labels:
            words_counter.update(option.split())

        clean_options = list()
        for option in options_labels:
            clean_options.append(option)
            for word, count in words_counter.most_common():
                if count == 1:
                    continue

                if word not in option:
                    continue

                if clean_options[-1].replace(word, "").strip() != "":
                    clean_options[-1] = clean_options[-1].replace(word, "").strip()

        return clean_options
