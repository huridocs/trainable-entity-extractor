from unittest import TestCase
from multilingual_paragraph_extractor.MultilingualParagraphExtractor import MultilingualParagraphExtractor
from multilingual_paragraph_extractor.domain.SegmentsFromLanguage import SegmentsFromLanguage
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.PdfDataSegment import PdfDataSegment


class TestMergeParagraphs(TestCase):
    extraction_identifier = ExtractionIdentifier(extraction_name="paragraph_extraction")

    def test_merge_paragraphs(self):
        pdf_data_segments_1 = PdfDataSegment.from_texts(texts=["English text"])
        language_segment_1 = SegmentsFromLanguage(language="en", segments=pdf_data_segments_1, is_main_language=True)

        pdf_data_segments_2 = PdfDataSegment.from_texts(texts=["French text"])
        language_segment_2 = SegmentsFromLanguage(language="fr", segments=pdf_data_segments_2, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphExtractor(extractor_identifier=self.extraction_identifier)
        segments_from_languages = [language_segment_1, language_segment_2]
        multilingual_paragraphs = multilingual_paragraph_extractor.extract_paragraphs(segments_from_languages)

        self.assertEqual(1, len(multilingual_paragraphs))
        self.assertEqual(2, len(multilingual_paragraphs[0].texts))
        self.assertEqual("en", multilingual_paragraphs[0].languages[0])
        self.assertEqual("fr", multilingual_paragraphs[0].languages[1])
        self.assertEqual("English text", multilingual_paragraphs[0].texts[0])
        self.assertEqual("French text", multilingual_paragraphs[0].texts[1])

    def test_merge_paragraphs_only_one_language(self):
        pdf_data_segments = PdfDataSegment.from_texts(texts=["English text", "English text too"])
        language_segment = SegmentsFromLanguage(language="en", segments=pdf_data_segments, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphExtractor(extractor_identifier=self.extraction_identifier)
        segments_from_languages = [language_segment]
        multilingual_paragraphs = multilingual_paragraph_extractor.extract_paragraphs(segments_from_languages)

        self.assertEqual(2, len(multilingual_paragraphs))
        self.assertEqual(1, len(multilingual_paragraphs[0].texts))
        self.assertEqual(1, len(multilingual_paragraphs[0].languages))
        self.assertEqual("en", multilingual_paragraphs[0].languages[0])
        self.assertEqual("en", multilingual_paragraphs[1].languages[0])
        self.assertEqual("English text", multilingual_paragraphs[0].texts[0])
        self.assertEqual("English text too", multilingual_paragraphs[1].texts[0])

    def test_merge_paragraphs_no_languages(self):
        multilingual_paragraph_extractor = MultilingualParagraphExtractor(extractor_identifier=self.extraction_identifier)
        segments_from_languages = []
        multilingual_paragraphs = multilingual_paragraph_extractor.extract_paragraphs(segments_from_languages)

        self.assertEqual(0, len(multilingual_paragraphs))

    def test_merge_paragraphs_when_no_main_language(self):
        pdf_data_segments_1 = PdfDataSegment.from_texts(texts=["English text"])
        language_segment_1 = SegmentsFromLanguage(language="en", segments=pdf_data_segments_1, is_main_language=False)

        pdf_data_segments_2 = PdfDataSegment.from_texts(texts=["French text"])
        language_segment_2 = SegmentsFromLanguage(language="fr", segments=pdf_data_segments_2, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphExtractor(extractor_identifier=self.extraction_identifier)
        segments_from_languages = [language_segment_1, language_segment_2]
        multilingual_paragraphs = multilingual_paragraph_extractor.extract_paragraphs(segments_from_languages)

        self.assertEqual(1, len(multilingual_paragraphs))
        self.assertEqual(2, len(multilingual_paragraphs[0].texts))
        self.assertEqual("en", multilingual_paragraphs[0].languages[0])
        self.assertEqual("fr", multilingual_paragraphs[0].languages[1])
        self.assertEqual("English text", multilingual_paragraphs[0].texts[0])
        self.assertEqual("French text", multilingual_paragraphs[0].texts[1])
