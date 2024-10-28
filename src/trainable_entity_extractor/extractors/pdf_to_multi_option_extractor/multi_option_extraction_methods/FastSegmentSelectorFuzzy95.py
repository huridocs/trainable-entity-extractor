import math
import unicodedata
from collections import Counter
from copy import deepcopy

from pdf_token_type_labels.TokenType import TokenType
from rapidfuzz import fuzz

from trainable_entity_extractor.data.Option import Option
from trainable_entity_extractor.data.PdfData import PdfData
from trainable_entity_extractor.data.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.data.TrainingSample import TrainingSample
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.PdfMultiOptionMethod import PdfMultiOptionMethod
from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.extractors.segment_selector.FastSegmentSelector import FastSegmentSelector
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyAll95 import (
    FuzzyAll95,
)


class FastSegmentSelectorFuzzy95(PdfMultiOptionMethod):
    threshold = 85

    text_types = [TokenType.TEXT, TokenType.LIST_ITEM, TokenType.TITLE, TokenType.SECTION_HEADER, TokenType.CAPTION]

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
        marked_segments = list()
        for sample in multi_option_data.samples:
            marked_segments.extend(self.get_marked_segments(sample))

        FastSegmentSelector(self.extraction_identifier, self.get_name()).create_model(marked_segments)

    def predict(self, multi_option_data: ExtractionData) -> list[list[Option]]:
        self.set_parameters(multi_option_data)
        self.extraction_data = self.get_prediction_data(multi_option_data)
        return FuzzyAll95().predict(self.extraction_data)

    def get_prediction_data(self, extraction_data: ExtractionData) -> ExtractionData:
        fast_segment_selector = FastSegmentSelector(self.extraction_identifier, self.get_name())
        predict_samples = list()
        for sample in extraction_data.samples:
            selected_segments = fast_segment_selector.predict(self.fix_two_pages_segments(sample))

            self.mark_segments_for_context(selected_segments)

            pdf_data = PdfData(None, file_name=sample.pdf_data.file_name)
            pdf_data.pdf_data_segments = selected_segments

            training_sample = TrainingSample(pdf_data=pdf_data, labeled_data=sample.labeled_data)
            predict_samples.append(training_sample)

        return ExtractionData(
            samples=predict_samples,
            options=self.extraction_data.options,
            multi_value=self.extraction_data.multi_value,
            extraction_identifier=self.extraction_identifier,
        )

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

    def get_marked_segments(self, training_sample: TrainingSample) -> list[PdfDataSegment]:
        cleaned_values = self.get_cleaned_options(training_sample.labeled_data.values)
        appearances_threshold = math.ceil(len(cleaned_values) * 0.68)

        if not appearances_threshold:
            return training_sample.pdf_data.pdf_data_segments

        fixed_segments = self.fix_two_pages_segments(training_sample)

        for segment in fixed_segments:
            appearances = len(self.get_appearances(segment, cleaned_values))

            if appearances_threshold <= appearances:
                segment.ml_label = 1

        return fixed_segments

    def fix_two_pages_segments(self, training_sample: TrainingSample) -> list[PdfDataSegment]:
        text_type_segments = [x for x in training_sample.pdf_data.pdf_data_segments if x.segment_type in self.text_types]

        fixed_segments = list()
        merged_segment = None
        for segment in training_sample.pdf_data.pdf_data_segments:
            if segment == merged_segment:
                merged_segment = None
                continue

            if not merged_segment:
                segment, merged_segment = FastSegmentSelectorFuzzy95.fix_segment(segment, text_type_segments)

            fixed_segments.append(segment)

        return fixed_segments

    @staticmethod
    def fix_segment(segment: PdfDataSegment, text_type_segments: list[PdfDataSegment]):
        if segment not in text_type_segments or not segment.text_content or segment.text_content[-1] != ",":
            return segment, None

        index = text_type_segments.index(segment)
        if index + 1 == len(text_type_segments) or segment.page_number >= text_type_segments[index + 1].page_number:
            return segment, None

        segment = deepcopy(segment)
        segment.text_content += " " + text_type_segments[index + 1].text_content

        return segment, text_type_segments[index + 1]

    @staticmethod
    def mark_segments_for_context(segments: list[PdfDataSegment]):
        for segment in segments:
            segment.ml_label = 1
