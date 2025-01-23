from unittest import TestCase

from pdf_token_type_labels.TokenType import TokenType

from multilingual_paragraph_extractor.MultilingualParagraphExtractor import MultilingualParagraphExtractor
from multilingual_paragraph_extractor.domain.SegmentsFromLanguage import SegmentsFromLanguage
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.PdfDataSegment import PdfDataSegment


class TestRemoveHeadersFooters(TestCase):
    extraction_identifier = ExtractionIdentifier(extraction_name="paragraph_extraction")

    def test_remove_headers_footers(self):
        pdf_data_segments_1 = PdfDataSegment.from_texts(texts=["PAGE_HEADER", "Text", "FOOTNOTE"])
        pdf_data_segments_1[0].segment_type = TokenType.PAGE_HEADER
        pdf_data_segments_1[2].segment_type = TokenType.FOOTNOTE
        language_segment_1 = SegmentsFromLanguage(language="en", segments=pdf_data_segments_1, is_main_language=True)

        pdf_data_segments_2 = PdfDataSegment.from_texts(texts=["Text", "PAGE_FOOTER"])
        pdf_data_segments_2[1].segment_type = TokenType.PAGE_FOOTER
        language_segment_2 = SegmentsFromLanguage(language="fr", segments=pdf_data_segments_2, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphExtractor(extractor_identifier=self.extraction_identifier)
        segments_from_languages = [language_segment_1, language_segment_2]
        multilingual_paragraphs = multilingual_paragraph_extractor.extract_paragraphs(segments_from_languages)

        self.assertEqual(1, len(multilingual_paragraphs))
        self.assertEqual(2, len(multilingual_paragraphs[0].texts))
        self.assertEqual(2, len(multilingual_paragraphs[0].languages))
        self.assertEqual("en", multilingual_paragraphs[0].languages[0])
        self.assertEqual("fr", multilingual_paragraphs[0].languages[1])
        self.assertEqual("Text", multilingual_paragraphs[0].texts[0])
        self.assertEqual("Text", multilingual_paragraphs[0].texts[1])
