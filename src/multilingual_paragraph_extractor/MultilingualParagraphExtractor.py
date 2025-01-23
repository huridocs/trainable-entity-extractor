from multilingual_paragraph_extractor.domain.SegmentsFromLanguage import SegmentsFromLanguage
from multilingual_paragraph_extractor.domain.MultilingualParagraph import MultilingualParagraph
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier


class MultilingualParagraphExtractor:
    def __init__(self, extractor_identifier: ExtractionIdentifier):
        self.extractor_identifier = extractor_identifier

    def is_same_paragraph(self, paragraph_text_1: str, paragraph_text_2: str) -> bool:
        return True

    def extract_paragraphs(self, segments_from_languages: list[SegmentsFromLanguage]) -> list[MultilingualParagraph]:
        main_language_segments = [
            language_segments for language_segments in segments_from_languages if language_segments.is_main_language
        ][0]

        multilingual_paragraphs: list[MultilingualParagraph] = []
        for pdf_data_segment in main_language_segments.segments:
            multilingual_paragraphs.append(
                MultilingualParagraph(languages=[main_language_segments.language], texts=[pdf_data_segment.text_content])
            )

        other_languages_segments = [
            language_segments for language_segments in segments_from_languages if not language_segments.is_main_language
        ]

        for multilingual_paragraph in multilingual_paragraphs:
            main_language_paragraph: str = multilingual_paragraph.texts[0]
            for language_segments in other_languages_segments:

                for pdf_data_segment in language_segments.segments:
                    if self.is_same_paragraph(main_language_paragraph, pdf_data_segment.text_content):
                        multilingual_paragraph.texts.append(pdf_data_segment.text_content)
                        multilingual_paragraph.languages.append(language_segments.language)

        return multilingual_paragraphs
