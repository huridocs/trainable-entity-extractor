import math
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


class FuzzyFirst(PdfMultiOptionMethod):
    @staticmethod
    def get_first_appearance(pdf_segments: list[PdfDataSegment], options: list[str]) -> Optional[Appearance]:
        for pdf_segment in pdf_segments:
            for ratio_threshold in range(100, 69, -10):
                for option in options:
                    if len(pdf_segment.text_content) < math.ceil(len(option) * ratio_threshold / 100):
                        continue
                    if fuzz.partial_ratio(option, pdf_segment.text_content.lower()) >= ratio_threshold:
                        pdf_segment.ml_label = 1
                        return Appearance(option_label=option, context=pdf_segment.text_content)

        return None

    def predict(self, multi_option_data: ExtractionData) -> list[list[Value]]:
        self.options = multi_option_data.options
        predictions: list[list[Value]] = list()
        options_labels = [x.label.lower() for x in multi_option_data.options]
        options_labels_sorted = list(sorted(options_labels, key=lambda x: len(x), reverse=True))
        for multi_option_sample in multi_option_data.samples:
            pdf_segments: list[PdfDataSegment] = [x for x in multi_option_sample.pdf_data.pdf_data_segments]
            appearance = self.get_first_appearance(pdf_segments, options_labels_sorted)
            if appearance:
                predictions.append([appearance.to_value(options_labels, self.options)])
            else:
                predictions.append([])

        return predictions

    def train(self, multi_option_data: ExtractionData):
        pass
