import math
import unicodedata
from collections import Counter
from copy import deepcopy
from typing import Type

from pdf_token_type_labels.TokenType import TokenType
from rapidfuzz import fuzz

from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.FilterSegmentsMethod import (
    FilterSegmentsMethod,
)
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.MultiLabelMethod import MultiLabelMethod
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.PdfMultiOptionMethod import (
    PdfMultiOptionMethod,
)
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.use_cases.extractors.segment_selector.FastSegmentSelector import FastSegmentSelector
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyAll95 import (
    FuzzyAll95,
)


class FastSegmentSelectorFuzzy95(PdfMultiOptionMethod):
    threshold = 85

    text_types = [TokenType.TEXT, TokenType.LIST_ITEM, TokenType.TITLE, TokenType.SECTION_HEADER, TokenType.CAPTION]

    def __init__(
        self, filter_segments_method: Type[FilterSegmentsMethod] = None, multi_label_method: Type[MultiLabelMethod] = None
    ):
        super().__init__(filter_segments_method, multi_label_method)
        self._remove_accents_cache = {}

    def get_appearances(self, pdf_segment: PdfDataSegment, options: list[str]) -> list[str]:
        appearances = []

        for option in options:
            if len(pdf_segment.text_content) < math.ceil(len(option)):
                continue

            if fuzz.partial_ratio(option, pdf_segment.text_content.lower()) >= self.threshold:
                appearances.append(option)

        return list(dict.fromkeys(appearances))

    def train(self, multi_option_data: ExtractionData):
        self.set_parameters(multi_option_data)

        if len(multi_option_data.samples) > 200:
            return

        marked_segments = list()
        for sample in multi_option_data.samples:
            marked_segments.extend(self.get_marked_segments(sample))
        FastSegmentSelector(self.extraction_identifier, self.get_name()).create_model(marked_segments)

    def predict(self, multi_option_data: ExtractionData) -> list[list[Value]]:
        self.set_parameters(multi_option_data)
        self.extraction_data = self.get_prediction_data(multi_option_data)
        predictions = FuzzyAll95().predict(self.extraction_data)
        return predictions

    def get_prediction_data(self, extraction_data: ExtractionData) -> ExtractionData:
        fast_segment_selector = FastSegmentSelector(self.extraction_identifier, self.get_name())
        predict_samples = list()
        for sample in extraction_data.samples:
            selected_segments = fast_segment_selector.predict(self.fix_two_pages_segments(sample))

            self.mark_segments_for_context(selected_segments)

            pdf_data = PdfData(file_name=sample.pdf_data.file_name)
            pdf_data.pdf_data_segments = selected_segments

            training_sample = TrainingSample(pdf_data=pdf_data, labeled_data=sample.labeled_data)
            predict_samples.append(training_sample)

        return ExtractionData(
            samples=predict_samples,
            options=self.extraction_data.options,
            multi_value=self.extraction_data.multi_value,
            extraction_identifier=self.extraction_identifier,
        )

    def remove_accents(self, text: str) -> str:
        if text in self._remove_accents_cache:
            return self._remove_accents_cache[text]

        nfkd_form = unicodedata.normalize("NFKD", text)
        only_ascii = nfkd_form.encode("ASCII", "ignore").decode("utf-8")
        self._remove_accents_cache[text] = only_ascii
        return only_ascii

    def get_cleaned_options(self, options: list[Option]) -> list[str]:
        options_labels = [self.remove_accents(x.label.lower()) for x in options]
        words_counter = Counter()
        for option_label in options_labels:
            words_counter.update(option_label.split())

        clean_options = []
        most_common_words = words_counter.most_common()
        for option_label in options_labels:
            cleaned_label = option_label
            for word, count in most_common_words:
                if count > 1 and word in cleaned_label and len(cleaned_label.replace(word, "").strip()) > 3:
                    cleaned_label = cleaned_label.replace(word, "").strip()
            clean_options.append(cleaned_label)

        return clean_options

    def get_marked_segments(self, training_sample: TrainingSample) -> list[PdfDataSegment]:
        cleaned_values = self.get_cleaned_options(training_sample.labeled_data.values)
        appearances_threshold = math.ceil(len(cleaned_values) * 0.68)

        if not appearances_threshold:
            return training_sample.pdf_data.pdf_data_segments

        fixed_segments = self.fix_two_pages_segments(training_sample)

        for segment in fixed_segments:
            if len(self.get_appearances(segment, cleaned_values)) >= appearances_threshold:
                segment.ml_label = 1

        return fixed_segments

    def fix_two_pages_segments(self, training_sample: TrainingSample) -> list[PdfDataSegment]:
        pdf_data_segments = training_sample.pdf_data.pdf_data_segments
        text_type_segments = [s for s in pdf_data_segments if s.segment_type in self.text_types]
        text_type_segments_set = set(text_type_segments)

        fixed_segments = []
        removed_segments = set()

        for segment in pdf_data_segments:
            if segment in removed_segments:
                continue

            new_segment, merged_segment = self._fix_segment(segment, text_type_segments, text_type_segments_set)
            fixed_segments.append(new_segment)
            if merged_segment is not None:
                removed_segments.add(merged_segment)

        return fixed_segments

    @staticmethod
    def _fix_segment(
        segment: PdfDataSegment, text_type_segments: list[PdfDataSegment], text_type_segments_set: set[PdfDataSegment]
    ):
        if segment in text_type_segments_set and segment.text_content and segment.text_content[-1] != ".":
            segment_index = text_type_segments.index(segment)
            if (
                segment_index + 1 < len(text_type_segments)
                and segment.page_number < text_type_segments[segment_index + 1].page_number
            ):
                new_segment = deepcopy(segment)
                new_segment.text_content += " " + text_type_segments[segment_index + 1].text_content
                return new_segment, text_type_segments[segment_index + 1]

        return segment, None

    @staticmethod
    def mark_segments_for_context(segments: list[PdfDataSegment]):
        for segment in segments:
            segment.ml_label = 1
