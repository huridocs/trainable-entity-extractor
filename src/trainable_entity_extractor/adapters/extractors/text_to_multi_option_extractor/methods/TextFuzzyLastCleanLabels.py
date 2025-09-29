import math

from rapidfuzz import fuzz

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.methods.TextFuzzyFirstCleanLabels import (
    TextFuzzyFirstCleanLabels,
)


class TextFuzzyLastCleanLabels(TextFuzzyFirstCleanLabels):

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        return True

    def get_appearance(self, texts: list[str], options: list[str]) -> list[str]:
        all_text = " ".join([self.remove_accents(text) for text in texts]).lower()
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
