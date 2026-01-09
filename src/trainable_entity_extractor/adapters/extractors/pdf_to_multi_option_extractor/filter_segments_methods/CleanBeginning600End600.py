from pdf_features import Rectangle

from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanBeginningDotDigits1000 import (
    CleanBeginningDotDigits1000,
)
from trainable_entity_extractor.domain.PdfDataSegment import PdfDataSegment


class CleanBeginning600End600(CleanBeginningDotDigits1000):
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

    def filter_segments(self, pdf_data_segments: list[PdfDataSegment]) -> list[PdfDataSegment]:
        tokens = self.get_first_tokens(pdf_data_segments, 600)
        remaining_segments = [x for x in pdf_data_segments if x.ml_label == 0]
        return tokens + self.get_last_tokens(remaining_segments, 600)
