from pdf_token_type_labels.TokenType import TokenType

from multilingual_paragraph_extractor.domain.SegmentsFromLanguage import SegmentsFromLanguage
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.PdfDataSegment import PdfDataSegment


class MultilingualParagraphExtractor:
    def __init__(self, extractor_identifier: ExtractionIdentifier):
        self.extractor_identifier = extractor_identifier

    def align_languages(self, segments_from_languages: list[SegmentsFromLanguage]):
        if not segments_from_languages:
            return []

        segments_from_languages = [self.remove_no_text_content(x) for x in segments_from_languages]
        segments_from_languages = [self.merge_segments_spanning_two_pages(x) for x in segments_from_languages]

        main_language, other_languages = self.get_main_and_other_languages(segments_from_languages)

        for language_segments in other_languages:
            self.align_language(main_language, language_segments)

    @staticmethod
    def get_main_and_other_languages(
        segments_from_languages: list[SegmentsFromLanguage],
    ) -> tuple[SegmentsFromLanguage, list[SegmentsFromLanguage]]:
        main_languages = [x for x in segments_from_languages if x.is_main_language]
        if not main_languages:
            return segments_from_languages[0], segments_from_languages[1:]

        main_language = main_languages[0]
        other_languages = [x for x in segments_from_languages if x != main_language]
        return main_language, other_languages

    @staticmethod
    def remove_no_text_content(segments_from_language: SegmentsFromLanguage) -> SegmentsFromLanguage:
        text_content_types = [
            TokenType.FORMULA,
            TokenType.LIST_ITEM,
            TokenType.TITLE,
            TokenType.TEXT,
            TokenType.SECTION_HEADER,
        ]
        segments = [x for x in segments_from_language.segments if x.segment_type in text_content_types]
        segments_from_language.segments = segments
        return segments_from_language

    def merge_segments_spanning_two_pages(self, segments_from_language: SegmentsFromLanguage) -> SegmentsFromLanguage:
        fixed_segments = []
        segments = segments_from_language.segments
        index = 0

        while index < len(segments):
            segment = segments[index]
            if index + 1 < len(segments) and self.are_same_segment_from_same_language(segment, segments[index + 1]):
                merged_segment = PdfDataSegment.from_list_to_merge([segment, segments[index + 1]])
                fixed_segments.append(merged_segment)
                index += 2
            else:
                fixed_segments.append(segment)
                index += 1

        segments_from_language.segments = fixed_segments
        return segments_from_language

    @staticmethod
    def are_same_segment_from_same_language(segment: PdfDataSegment, next_segment: PdfDataSegment) -> bool:
        if segment.page_number == next_segment.page_number:
            return False

        if int(segment.page_number - next_segment.page_number) > 1:
            return False

        if segment.segment_type != next_segment.segment_type:
            return False

        if segment.text_content[-1] in [".", "!", "?", ";"]:
            return False

        if next_segment.text_content[0].isupper():
            return False

        if next_segment.text_content[0].isdigit():
            return False

        return True

    def align_language(self, main_language: SegmentsFromLanguage, segments_from_language: SegmentsFromLanguage):
        for index, main_segment in enumerate(main_language.segments):
            segments_to_align = segments_from_language.segments

            segment = None if index >= len(segments_to_align) else segments_to_align[index]

            if not segment:
                segments_to_align.append(PdfDataSegment.from_text(""))
                continue

            if main_segment.are_similar(segment):
                continue

            main_next_segment = None if index + 1 >= len(main_language.segments) else main_language.segments[index + 1]

            if main_next_segment and main_next_segment.are_similar(segment):
                segments_to_align.insert(index, PdfDataSegment.from_text(""))
