import re
from copy import deepcopy

from pdf_features.Rectangle import Rectangle

from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FastSegmentSelectorFuzzyCommas import (
    FastSegmentSelectorFuzzyCommas,
)


class SentenceSelectorFuzzyCommas(FastSegmentSelectorFuzzyCommas):
    def train(self, multi_option_data: ExtractionData):
        self.set_parameters(multi_option_data)
        extraction_data_by_sentences = self.get_extraction_data_by_sentence(multi_option_data)
        super().train(extraction_data_by_sentences)

    def predict(self, multi_option_data: ExtractionData) -> list[list[Value]]:
        extraction_data_by_sentences = self.get_extraction_data_by_sentence(multi_option_data)
        return super().predict(extraction_data_by_sentences)

    def get_extraction_data_by_sentence(self, multi_option_data: ExtractionData) -> ExtractionData:
        samples_by_sentence = []
        for sample in multi_option_data.samples:
            sentence_segment_list = self.get_sentence_segment_list(sample.pdf_data.pdf_data_segments)
            samples_by_sentence.append(self.get_sample(sample, sentence_segment_list))

        return ExtractionData(
            samples=samples_by_sentence,
            options=multi_option_data.options,
            multi_value=multi_option_data.multi_value,
            extraction_identifier=multi_option_data.extraction_identifier,
        )

    def get_sentence_segment_list(self, pdf_data_segments) -> list[(str, PdfDataSegment)]:
        text_segments = [segment for segment in pdf_data_segments if segment.segment_type in self.text_types]
        merged_sentences = self.get_segments_merged(text_segments)

        sentence_segment_list = []
        for segment in merged_sentences:
            segment_text = " ".join(segment.text_content.split())
            for text in re.split(r"\.|:", segment_text):
                if not text:
                    continue

                sentence_segment_list.append((text, segment))

        if not sentence_segment_list:
            return list()

        sentences_across_pages = list()
        sentences_across_pages.append(sentence_segment_list[0])
        for sentence, next_sentence in zip(sentence_segment_list, sentence_segment_list[1:]):
            if sentence[0][-1] == ",":
                merged_sentences = " ".join([sentences_across_pages[-1][0], next_sentence[0]])
                sentences_across_pages[-1] = (merged_sentences, sentences_across_pages[-1][1])
                continue

            sentences_across_pages.append(next_sentence)

        return sentences_across_pages

    @staticmethod
    def get_segments_merged(segments):
        segments = [segment for segment in segments if segment.text_content.strip()]
        if not segments:
            return list()

        merged_sentences = [segments[0]]
        for segment in segments[1:]:
            previous_segment_text = " ".join(merged_sentences[-1].text_content.split())

            if previous_segment_text[-1] not in [".", ":"]:
                merged_segment = deepcopy(merged_sentences[-1])
                merged_segment.text_content = f"{previous_segment_text}, {' '.join(segment.text_content.split())}"
                bounding_boxes = [merged_segment.bounding_box, segment.bounding_box]
                merged_segment.bounding_box = Rectangle.merge_rectangles(bounding_boxes)
                merged_sentences[-1] = merged_segment
                continue

            merged_sentences.append(segment)
        return merged_sentences

    @staticmethod
    def get_sample(sample: TrainingSample, sentence_segment_list: list[(str, PdfDataSegment)]) -> TrainingSample:
        sentence_segments = list()
        for sentence, segment in sentence_segment_list:
            sentence_segment = deepcopy(segment)
            sentence_segment.text_content = sentence
            sentence_segments.append(sentence_segment)

        sentence_pdf_data = PdfData(file_name=sample.pdf_data.file_name)
        sentence_pdf_data.pdf_data_segments = sentence_segments
        return TrainingSample(pdf_data=sentence_pdf_data, labeled_data=sample.labeled_data)
