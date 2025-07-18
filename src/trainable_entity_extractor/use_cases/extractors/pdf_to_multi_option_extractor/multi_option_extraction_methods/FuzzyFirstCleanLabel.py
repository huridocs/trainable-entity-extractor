import math
import unicodedata
from collections import Counter
from typing import Optional

from rapidfuzz import fuzz

from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.PdfMultiOptionMethod import (
    PdfMultiOptionMethod,
)

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.Appearance import (
    Appearance,
)


class FuzzyFirstCleanLabel(PdfMultiOptionMethod):

    def get_appearance(self, pdf_segments: list[PdfDataSegment], options: list[str]) -> Optional[Appearance]:
        for pdf_segment in pdf_segments:
            for ratio_threshold in range(100, 95, -1):
                for option in options:
                    if len(pdf_segment.text_content) < math.ceil(len(option) * ratio_threshold / 100):
                        continue
                    text = self.remove_accents(pdf_segment.text_content.lower())
                    if fuzz.partial_ratio(option, text) >= ratio_threshold:
                        return Appearance(option_label=option, context=pdf_segment.text_content)

        return None

    def predict(self, multi_option_data: ExtractionData) -> list[list[Value]]:
        self.options = multi_option_data.options
        predictions: list[list[Value]] = list()
        clean_options = self.get_cleaned_options(multi_option_data.options)
        clean_options_sorted = list(sorted(clean_options, key=lambda x: len(x), reverse=True))

        for multi_option_sample in multi_option_data.samples:
            pdf_segments: list[PdfDataSegment] = [x for x in multi_option_sample.pdf_data.pdf_data_segments]
            appearance = self.get_appearance(pdf_segments, clean_options_sorted)
            if appearance:
                predictions.append([appearance.to_value(clean_options, self.options)])
            else:
                predictions.append([])

        return predictions

    def train(self, multi_option_data: ExtractionData):
        pass

    @staticmethod
    def remove_accents(text: str):
        nfkd_form = unicodedata.normalize("NFKD", text)
        only_ascii = nfkd_form.encode("ASCII", "ignore")
        return only_ascii.decode()

    def get_cleaned_options(self, options: list[Option]) -> list[str]:
        options_labels = [self.remove_accents(x.label.lower()) for x in options]
        words_counter = Counter()
        for option_label in options_labels:
            words_counter.update(option_label.split())

        clean_options = list()
        for option_label in options_labels:
            clean_options.append(option_label)
            for word, count in words_counter.most_common():
                if count == 1:
                    continue

                if word not in option_label:
                    continue

                if len(clean_options[-1].replace(word, "").strip()) > 3:
                    clean_options[-1] = clean_options[-1].replace(word, "").strip()

        return clean_options
