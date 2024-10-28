from trainable_entity_extractor.data.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.filter_segments_methods.Beginning750 import (
    Beginning750,
)


class End750(Beginning750):
    def get_first_tokens(self, pdf_data_segments: list[PdfDataSegment], text_length: int) -> list[PdfDataSegment]:
        total_text = ""
        filtered_segments: list[PdfDataSegment] = list()
        for pdf_data_segment in reversed(pdf_data_segments):
            pdf_data_segment_copy = self.get_segment(pdf_data_segment, text_length - len(total_text))

            if not pdf_data_segment_copy:
                break

            total_text += " " + pdf_data_segment_copy.text_content
            filtered_segments.append(pdf_data_segment_copy)

        return list(reversed(filtered_segments))
