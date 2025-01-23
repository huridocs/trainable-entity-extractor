from pdf_token_type_labels.TokenType import TokenType

from multilingual_paragraph_extractor.domain.SegmentsFromLanguage import SegmentsFromLanguage
from multilingual_paragraph_extractor.domain.MultilingualParagraph import MultilingualParagraph
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.PdfDataSegment import PdfDataSegment


class MultilingualParagraphExtractor:
    def __init__(self, extractor_identifier: ExtractionIdentifier):
        self.extractor_identifier = extractor_identifier

    def extract_paragraphs(self, segments_from_languages: list[SegmentsFromLanguage]) -> list[MultilingualParagraph]:
        if not segments_from_languages:
            return []

        segments_from_languages = self.remove_headers_footers(segments_from_languages)
        multilingual_paragraphs: list[MultilingualParagraph] = []

        main_language, other_languages = self.get_main_and_other_languages(segments_from_languages)
        for data_segment in main_language.segments:
            paragraph = MultilingualParagraph(languages=[main_language.language], texts=[data_segment.text_content])
            multilingual_paragraphs.append(paragraph)

        for language_segments in other_languages:
            for i, data_segment in enumerate(language_segments.segments):
                multilingual_paragraphs[i].texts.append(data_segment.text_content)
                multilingual_paragraphs[i].languages.append(language_segments.language)

        return multilingual_paragraphs

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

    def remove_headers_footers(self, segments_from_languages: list[SegmentsFromLanguage]) -> list[SegmentsFromLanguage]:
        return [
            SegmentsFromLanguage(
                language=segments_from_language.language,
                segments=[x for x in segments_from_language.segments if not self.is_header_or_footer(x)],
                is_main_language=segments_from_language.is_main_language,
            )
            for segments_from_language in segments_from_languages
        ]

    @staticmethod
    def is_header_or_footer(pdf_data_segment: PdfDataSegment) -> bool:
        return pdf_data_segment.segment_type in [TokenType.PAGE_HEADER, TokenType.PAGE_FOOTER, TokenType.FOOTNOTE]
