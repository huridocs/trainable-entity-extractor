import math
from typing import Optional

from rapidfuzz import fuzz

from trainable_entity_extractor.domain.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.PdfMultiOptionMethod import (
    PdfMultiOptionMethod,
)

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.Appearance import (
    Appearance,
)


class FuzzyAll100(PdfMultiOptionMethod):

    threshold = 100

    def get_appearances(self, pdf_segments: list[PdfDataSegment], options: list[str]) -> list[Appearance]:
        appearances: list[Appearance] = []
        for pdf_segment in pdf_segments:
            text = " ".join(pdf_segment.text_content.lower().split())
            for option in options:
                if option in appearances:
                    continue

                if len(text) < math.ceil(len(option) * self.threshold / 100):
                    continue

                if fuzz.partial_ratio(option, text) >= self.threshold:
                    pdf_segment.ml_label = 1
                    appearances.append(Appearance(option_label=option, context=pdf_segment.text_content))

                if option in text:
                    text = text.replace(option, "")

        return appearances

    def predict(self, prediction_samples_data: PredictionSamplesData) -> list[list[Value]]:
        self.options = prediction_samples_data.options
        self.multi_value = prediction_samples_data.multi_value
        predictions = list()
        options_labels = [x.label.lower() for x in prediction_samples_data.options]
        options_labels_sorted = list(sorted(options_labels, key=lambda x: len(x), reverse=True))
        for prediction_sample in prediction_samples_data.prediction_samples:
            pdf_segments: list[PdfDataSegment] = [x for x in prediction_sample.pdf_data.pdf_data_segments]
            appearances = self.get_appearances(pdf_segments, options_labels_sorted)
            values = [x.to_value(options_labels, prediction_samples_data.options) for x in appearances]
            predictions.append(values)

        return predictions

    def train(self, multi_option_data: ExtractionData):
        pass
