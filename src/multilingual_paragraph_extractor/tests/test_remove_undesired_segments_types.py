from unittest import TestCase

from pdf_token_type_labels.TokenType import TokenType

from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures
from multilingual_paragraph_extractor.domain.SegmentsFromLanguage import ParagraphsFromLanguage
from multilingual_paragraph_extractor.use_cases.MultilingualParagraphAlignerUseCase import (
    MultilingualParagraphAlignerUseCase,
)
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier


class TestRemoveUndesiredSegmentTypes(TestCase):
    extraction_identifier = ExtractionIdentifier(extraction_name="paragraph_extraction")

    def test_remove_undesired_segment_types(self):
        segments = ParagraphFeatures.from_texts(texts=["PAGE_HEADER", "Text", "FOOTNOTE"])
        segments[0].segment_type = TokenType.PAGE_HEADER
        segments[2].segment_type = TokenType.FOOTNOTE
        language_segment_1 = ParagraphsFromLanguage(language="en", segments=segments, is_main_language=True)

        other_segments = ParagraphFeatures.from_texts(texts=["Text", "PAGE_FOOTER", "PICTURE", "CAPTION"])
        other_segments[1].segment_type = TokenType.PAGE_FOOTER
        other_segments[2].segment_type = TokenType.PICTURE
        other_segments[3].segment_type = TokenType.CAPTION
        language_segment_2 = ParagraphsFromLanguage(language="fr", segments=other_segments, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        segments_from_languages = [language_segment_1, language_segment_2]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(2, len(segments_from_languages))
        self.assertEqual(1, len(segments_from_languages[0].paragraphs))
        self.assertEqual(1, len(segments_from_languages[1].paragraphs))
        self.assertEqual("en", segments_from_languages[0].language)
        self.assertEqual("fr", segments_from_languages[1].language)
        self.assertEqual("Text", segments_from_languages[0].paragraphs[0].text_content)
        self.assertEqual("Text", segments_from_languages[1].paragraphs[0].text_content)
