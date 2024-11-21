import re

from trainable_entity_extractor.data.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.extractors.pdf_to_text_extractor.methods.FirstDateMethod import FirstDateMethod
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.GlinerDateParserMethod import (
    GlinerDateParserMethod,
)


class GlinerFirstDateMethod(FirstDateMethod):
    @staticmethod
    def contains_year(text: str):
        year_pattern = re.compile(r"([0-9]{2})")
        return bool(year_pattern.search(text.replace(" ", "")))

    def get_date_from_segments(self, segments: list[PdfDataSegment], languages):
        merge_segments: list[list[PdfDataSegment]] = self.merge_segments_for_dates(segments)
        for segments in merge_segments:
            segment_merged = PdfDataSegment.from_list_to_merge(segments)
            if not self.contains_year(segment_merged.text_content):
                continue

            date = GlinerDateParserMethod.get_date([segment_merged.text_content])
            if date:
                for segment in segments:
                    segment.ml_label = 1
                return date.strftime("%Y-%m-%d")

        return ""

    def merge_segments_for_dates(self, segments: list[PdfDataSegment]):
        min_words = 35
        merge_segments: list[list[PdfDataSegment]] = list()
        for segment in segments:
            if not merge_segments:
                merge_segments.append([segment])
                continue

            words_previous_segment = self.count_segments_words(merge_segments[-1])

            if words_previous_segment < min_words:
                merge_segments[-1].append(segment)

            merge_segments.append([segment])

        return merge_segments

    @staticmethod
    def count_segments_words(segments: list[PdfDataSegment]):
        return sum([len(segment.text_content.split()) for segment in segments])
