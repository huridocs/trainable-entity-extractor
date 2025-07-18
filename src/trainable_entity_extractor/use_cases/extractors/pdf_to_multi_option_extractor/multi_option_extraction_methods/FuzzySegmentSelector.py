import math
import unicodedata
from collections import Counter

from rapidfuzz import fuzz

from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.PdfMultiOptionMethod import (
    PdfMultiOptionMethod,
)
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.Appearance import (
    Appearance,
)
from trainable_entity_extractor.use_cases.extractors.segment_selector.SegmentSelector import SegmentSelector

threshold = 85


class FuzzySegmentSelector(PdfMultiOptionMethod):
    @staticmethod
    def get_appearances(pdf_segments: list[PdfDataSegment], options: list[str]) -> list[Appearance]:
        appearances = []

        for pdf_segment in pdf_segments:
            for option in options:
                if option in appearances:
                    continue

                if len(pdf_segment.text_content) < math.ceil(len(option)):
                    continue

                if fuzz.partial_ratio(option, pdf_segment.text_content.lower()) >= threshold:
                    appearances.append(Appearance(option_label=option, context=pdf_segment.text_content))

        return appearances

    def predict(self, multi_option_data: ExtractionData) -> list[list[Value]]:
        self.options = multi_option_data.options
        segment_selector = SegmentSelector(self.extraction_identifier)
        segment_selector.set_extraction_segments([sample.pdf_data for sample in multi_option_data.samples])

        predictions = list()
        clean_options = self.get_cleaned_options(multi_option_data.options)
        for multi_option_sample in multi_option_data.samples:
            pdf_segments: list[PdfDataSegment] = [x for x in multi_option_sample.pdf_data.pdf_data_segments if x.ml_label]
            appearances: list[Appearance] = self.get_appearances(pdf_segments, clean_options)
            predictions.append([x.to_value(clean_options, self.options) for x in appearances])

        return predictions

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

    def get_segments_appearances(
        self, pdf_data_segments: list[PdfDataSegment], segment_index: int, cleaned_options: list[str]
    ) -> (PdfDataSegment, int, PdfDataSegment, int):
        segment = pdf_data_segments[segment_index]
        next_segment = self.get_next_segment(pdf_data_segments, segment_index)
        appearances = len(self.get_appearances(segment, cleaned_options))

        if next_segment:
            next_segment_appearances = len(self.get_appearances(next_segment, cleaned_options))
        else:
            next_segment_appearances = 0

        return appearances, next_segment, next_segment_appearances

    def train(self, multi_option_data: ExtractionData):
        self.set_parameters(multi_option_data)
        for multi_option_sample in multi_option_data.samples:
            self.mark_segments_for_segment_selector(multi_option_sample)

        segment_selector = SegmentSelector(self.extraction_identifier)
        pdfs_data = [sample.pdf_data for sample in multi_option_data.samples]
        segment_selector.create_model(pdfs_data=pdfs_data)

    def mark_segments_for_segment_selector(self, multi_option_sample: TrainingSample):
        cleaned_values = self.get_cleaned_options(multi_option_sample.labeled_data.values)
        appearances_threshold = math.ceil(len(cleaned_values) * threshold / 100)

        if not appearances_threshold:
            return

        for i, segment in enumerate(multi_option_sample.pdf_data.pdf_data_segments):
            appearances, next_segment, next_segment_appearances = self.get_segments_appearances(
                multi_option_sample.pdf_data.pdf_data_segments, i, cleaned_values
            )

            if next_segment_appearances and appearances_threshold <= appearances + next_segment_appearances:
                segment.ml_label = 1
                next_segment.ml_label = 1
                break

            if appearances_threshold <= appearances:
                segment.ml_label = 1
                break

    @staticmethod
    def get_next_segment(pdf_data_segments: list[PdfDataSegment], segment_index: int):
        for j in range(segment_index + 1, len(pdf_data_segments)):
            if pdf_data_segments[j].segment_type == pdf_data_segments[segment_index].segment_type:
                return pdf_data_segments[j]

        return None
