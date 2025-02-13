from unittest import TestCase

from pdf_token_type_labels.TokenType import TokenType

from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures
from multilingual_paragraph_extractor.domain.ParagraphsFromLanguage import ParagraphsFromLanguage
from multilingual_paragraph_extractor.use_cases.MultilingualParagraphAlignerUseCase import (
    MultilingualParagraphAlignerUseCase,
)
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier


class TestRemoveUndesiredSegmentTypes(TestCase):
    extraction_identifier = ExtractionIdentifier(extraction_name="paragraph_extraction")

    def test_remove_undesired_segment_types(self):
        paragraphs = ParagraphFeatures.from_texts(texts=["PAGE_HEADER", "Text en", "FOOTNOTE"])
        paragraphs[0].paragraph_type = TokenType.PAGE_HEADER
        paragraphs[2].paragraph_type = TokenType.FOOTNOTE
        language_segment_1 = ParagraphsFromLanguage(language="en", paragraphs=paragraphs, is_main_language=True)

        other_paragraphs = ParagraphFeatures.from_texts(texts=["Text fr", "PAGE_FOOTER", "PICTURE", "CAPTION"])
        other_paragraphs[1].paragraph_type = TokenType.PAGE_FOOTER
        other_paragraphs[2].paragraph_type = TokenType.PICTURE
        other_paragraphs[3].paragraph_type = TokenType.CAPTION
        language_segment_2 = ParagraphsFromLanguage(language="fr", paragraphs=other_paragraphs, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        segments_from_languages = [language_segment_1, language_segment_2]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(2, len(segments_from_languages))
        self.assertEqual(1, len(segments_from_languages[0]._aligned_paragraphs))
        self.assertEqual(1, len(segments_from_languages[1]._aligned_paragraphs))
        self.assertEqual("en", segments_from_languages[0].language)
        self.assertEqual("fr", segments_from_languages[1].language)
        self.assertEqual("Text en", segments_from_languages[0]._aligned_paragraphs[0].text_cleaned)
        self.assertEqual("Text fr", segments_from_languages[1]._aligned_paragraphs[0].text_cleaned)
