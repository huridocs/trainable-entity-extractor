from unittest import TestCase
from multilingual_paragraph_extractor.use_cases.MultilingualParagraphExtractor import MultilingualParagraphExtractor
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
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(2, len(segments_from_languages))
        self.assertEqual(1, len(segments_from_languages[0].segments))
        self.assertEqual(1, len(segments_from_languages[1].segments))

        self.assertEqual("en", segments_from_languages[0].language)
        self.assertEqual("fr", segments_from_languages[1].language)

        self.assertEqual("English text", segments_from_languages[0].segments[0].text_content)
        self.assertEqual("French text", segments_from_languages[1].segments[0].text_content)

    def test_merge_paragraphs_only_one_language(self):
        pdf_data_segments = PdfDataSegment.from_texts(texts=["English text", "English text too"])
        language_segment = SegmentsFromLanguage(language="en", segments=pdf_data_segments, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphExtractor(extractor_identifier=self.extraction_identifier)
        segments_from_languages = [language_segment]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(1, len(segments_from_languages))
        self.assertEqual(2, len(segments_from_languages[0].segments))

        self.assertEqual("en", segments_from_languages[0].language)
        self.assertEqual("English text", segments_from_languages[0].segments[0].text_content)
        self.assertEqual("English text too", segments_from_languages[0].segments[1].text_content)

    def test_merge_paragraphs_no_languages(self):
        multilingual_paragraph_extractor = MultilingualParagraphExtractor(extractor_identifier=self.extraction_identifier)
        segments_from_languages = []
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(0, len(segments_from_languages))

    def test_merge_paragraphs_when_no_main_language(self):
        pdf_data_segments_1 = PdfDataSegment.from_texts(texts=["English text"])
        language_segment_1 = SegmentsFromLanguage(language="en", segments=pdf_data_segments_1, is_main_language=False)

        pdf_data_segments_2 = PdfDataSegment.from_texts(texts=["French text"])
        language_segment_2 = SegmentsFromLanguage(language="fr", segments=pdf_data_segments_2, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphExtractor(extractor_identifier=self.extraction_identifier)
        segments_from_languages = [language_segment_1, language_segment_2]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(2, len(segments_from_languages))
        self.assertEqual(1, len(segments_from_languages[0].segments))
        self.assertEqual(1, len(segments_from_languages[1].segments))

        self.assertEqual("en", segments_from_languages[0].language)
        self.assertEqual("fr", segments_from_languages[1].language)

        self.assertEqual("English text", segments_from_languages[0].segments[0].text_content)
        self.assertEqual("French text", segments_from_languages[1].segments[0].text_content)
