import math

from rapidfuzz import fuzz

from trainable_entity_extractor.data.Option import Option
from trainable_entity_extractor.data.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.PdfMultiOptionMethod import PdfMultiOptionMethod

from trainable_entity_extractor.data.ExtractionData import ExtractionData


class FuzzyAll100(PdfMultiOptionMethod):

    threshold = 100

    def get_appearances(self, pdf_segments: list[PdfDataSegment], options: list[str]) -> list[str]:
        appearances = []
        for pdf_segment in pdf_segments:
            text = " ".join(pdf_segment.text_content.lower().split())
            for option in options:
                if len(text) < math.ceil(len(option) * self.threshold / 100):
                    continue

                if fuzz.partial_ratio(option, text) >= self.threshold:
                    pdf_segment.ml_label = 1
                    appearances.append(option)

                if option in text:
                    text = text.replace(option, "")

        return list(dict.fromkeys(appearances))

    def predict(self, multi_option_data: ExtractionData) -> list[list[Option]]:
        predictions = list()
        options_labels = [x.label.lower() for x in multi_option_data.options]
        options_labels_sorted = list(sorted(options_labels, key=lambda x: len(x), reverse=True))
        for multi_option_sample in multi_option_data.samples:
            pdf_segments: list[PdfDataSegment] = [x for x in multi_option_sample.pdf_data.pdf_data_segments]
            predictions_sample = self.get_appearances(pdf_segments, options_labels_sorted)
            predictions.append([multi_option_data.options[options_labels.index(x)] for x in predictions_sample])

        return predictions

    def train(self, multi_option_data: ExtractionData):
        pass
