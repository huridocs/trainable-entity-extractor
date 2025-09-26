from copy import deepcopy

from pdf_features.Rectangle import Rectangle

from trainable_entity_extractor.domain.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.FilterSegmentsMethod import (
    FilterSegmentsMethod,
)


class CleanEndDot250(FilterSegmentsMethod):
    def get_last_tokens(self, pdf_data_segments: list[PdfDataSegment], text_length: int) -> list[PdfDataSegment]:
        total_text = ""
        filtered_segments: list[PdfDataSegment] = list()
        for pdf_data_segment in reversed(pdf_data_segments):
            pdf_data_segment_copy = self.clean_content_pdf_token(pdf_data_segment, text_length - len(total_text))

            if not pdf_data_segment_copy:
                break

            if pdf_data_segment_copy.text_content and "." == pdf_data_segment.text_content[-1]:
                pdf_data_segment_copy.text_content += "."

            total_text += " " + pdf_data_segment_copy.text_content
            filtered_segments.append(pdf_data_segment_copy)

        if not pdf_data_segments or "".join([x.text_content.strip() for x in filtered_segments]) == "":
            return [PdfDataSegment.from_values(1, Rectangle.from_coordinates(0, 0, 0, 0), "no text")]

        return list(reversed(filtered_segments))

    @staticmethod
    def clean_content_pdf_token(pdf_data_segment: PdfDataSegment, character_limit: int):
        if character_limit <= 0:
            return None

        pdf_data_segment.ml_label = 1
        pdf_data_segment_copy = deepcopy(pdf_data_segment)
        words = list()
        text = ""
        for word in pdf_data_segment_copy.text_content.split():
            clean_word = "".join([x for x in word if x.isalpha()])

            if len(text + " " + clean_word) > character_limit:
                break

            if clean_word:
                words.append(clean_word)
                text += " " + word

        pdf_data_segment_copy.text_content = " ".join(words)
        return pdf_data_segment_copy

    def filter_segments(self, pdf_data_segments: list[PdfDataSegment]) -> list[PdfDataSegment]:
        return self.get_last_tokens(pdf_data_segments, 250)
