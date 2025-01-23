from unittest import TestCase
from multilingual_paragraph_extractor.MultilingualParagraphExtractor import MultilingualParagraphExtractor
from multilingual_paragraph_extractor.domain.SegmentsFromLanguage import SegmentsFromLanguage
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.PdfDataSegment import PdfDataSegment


class TestMergeParagraphs(TestCase):

    def test_merge_paragraphs(self):
        pdf_data_segments = PdfDataSegment.from_texts(texts=["English text"])
        pdf_data_segments_2 = PdfDataSegment.from_texts(texts=["French text"])
        language_segment_1 = SegmentsFromLanguage(language="en", segments=pdf_data_segments, is_main_language=True)
        language_segment_2 = SegmentsFromLanguage(language="fr", segments=pdf_data_segments_2, is_main_language=False)
        extraction_identifier = ExtractionIdentifier(extraction_name="paragraph_extraction")
        multilingual_paragraphs = MultilingualParagraphExtractor(extractor_identifier=extraction_identifier).extract_paragraphs(
            [language_segment_1, language_segment_2]
        )

        self.assertEqual(1, len(multilingual_paragraphs))
        self.assertEqual(2, len(multilingual_paragraphs[0].texts))
        self.assertEqual("en", multilingual_paragraphs[0].languages[0])
        self.assertEqual("fr", multilingual_paragraphs[0].languages[1])
        self.assertEqual("English text", multilingual_paragraphs[0].texts[0])
        self.assertEqual("French text", multilingual_paragraphs[0].texts[1])
