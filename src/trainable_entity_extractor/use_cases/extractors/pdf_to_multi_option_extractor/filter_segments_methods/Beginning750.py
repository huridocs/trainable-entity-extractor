from copy import deepcopy

from trainable_entity_extractor.domain.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.FilterSegmentsMethod import (
    FilterSegmentsMethod,
)


class Beginning750(FilterSegmentsMethod):
    def get_first_tokens(self, pdf_data_segments: list[PdfDataSegment], text_length: int) -> list[PdfDataSegment]:
        total_text = ""
        filtered_segments: list[PdfDataSegment] = list()
        for pdf_data_segment in pdf_data_segments:
            pdf_data_segment_copy = self.get_segment(pdf_data_segment, text_length - len(total_text))

            if not pdf_data_segment_copy:
                break

            total_text += " " + pdf_data_segment_copy.text_content
            filtered_segments.append(pdf_data_segment_copy)

        return filtered_segments

    @staticmethod
    def get_segment(pdf_data_segment: PdfDataSegment, character_limit: int):
        if character_limit <= 0:
            return None

        pdf_data_segment_copy = deepcopy(pdf_data_segment)

        words = list()
        text = ""
        for word in pdf_data_segment_copy.text_content.split():
            if len(text + " " + word) > character_limit:
                break

            words.append(word)
            text += " " + word

        pdf_data_segment_copy.text_content = " ".join(words)
        return pdf_data_segment_copy

    def filter_segments(self, pdf_data_segments: list[PdfDataSegment]) -> list[PdfDataSegment]:
        return self.get_first_tokens(pdf_data_segments, 750)
