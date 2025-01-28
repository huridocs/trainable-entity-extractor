from unittest import TestCase

from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType

from multilingual_paragraph_extractor.use_cases.MultilingualParagraphExtractor import MultilingualParagraphExtractor
from multilingual_paragraph_extractor.domain.SegmentsFromLanguage import SegmentsFromLanguage
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.PdfDataSegment import PdfDataSegment


class TestMergeSegmentsSpanningTwoPages(TestCase):
    extraction_identifier = ExtractionIdentifier(extraction_name="paragraph_extraction")

    @staticmethod
    def get_segments():
        regular_segment_1 = PdfDataSegment(1, Rectangle(0, 0, 0, 0), "Text.")
        beginning_segments = PdfDataSegment(1, Rectangle(0, 100, 100, 200), "Text to be continued")
        end_segments = PdfDataSegment(2, Rectangle(0, 0, 0, 0), "here")
        regular_segment_2 = PdfDataSegment(2, Rectangle(0, 100, 100, 200), "Text.")

        return regular_segment_1, beginning_segments, end_segments, regular_segment_2

    def test_merge_segments_spanning_two_pages(self):
        regular_segment_1, beginning_segments, end_segments, regular_segment_2 = self.get_segments()

        segments = [regular_segment_1, beginning_segments, end_segments, regular_segment_2]
        language_segment = SegmentsFromLanguage(language="en", segments=segments, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphExtractor(extractor_identifier=self.extraction_identifier)
        segments_from_languages = [language_segment]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(3, len(segments_from_languages[0].segments))
        self.assertEqual("Text.", segments_from_languages[0].segments[0].text_content)
        self.assertEqual("Text to be continued here", segments_from_languages[0].segments[1].text_content)
        self.assertEqual("Text.", segments_from_languages[0].segments[2].text_content)

    def test_not_merge_when_same_page(self):
        regular_segment_1, beginning_segments, end_segments, regular_segment_2 = self.get_segments()

        regular_segment_1.page_number = 5
        beginning_segments.page_number = 5
        end_segments.page_number = 5
        regular_segment_2.page_number = 5

        segments = [regular_segment_1, beginning_segments, end_segments, regular_segment_2]
        language_segment = SegmentsFromLanguage(language="en", segments=segments, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphExtractor(extractor_identifier=self.extraction_identifier)
        segments_from_languages = [language_segment]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(4, len(segments_from_languages[0].segments))
        self.assertEqual("Text.", segments_from_languages[0].segments[0].text_content)
        self.assertEqual("Text to be continued", segments_from_languages[0].segments[1].text_content)
        self.assertEqual("here", segments_from_languages[0].segments[2].text_content)
        self.assertEqual("Text.", segments_from_languages[0].segments[3].text_content)

    def test_not_merge_when_ends_with_dot(self):
        regular_segment_1, beginning_segments, end_segments, regular_segment_2 = self.get_segments()

        beginning_segments.text_content = "Text not to be continued."

        segments = [regular_segment_1, beginning_segments, end_segments, regular_segment_2]
        language_segment = SegmentsFromLanguage(language="en", segments=segments, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphExtractor(extractor_identifier=self.extraction_identifier)
        segments_from_languages = [language_segment]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(4, len(segments_from_languages[0].segments))
        self.assertEqual("Text.", segments_from_languages[0].segments[0].text_content)
        self.assertEqual("Text not to be continued.", segments_from_languages[0].segments[1].text_content)
        self.assertEqual("here", segments_from_languages[0].segments[2].text_content)
        self.assertEqual("Text.", segments_from_languages[0].segments[3].text_content)

    def test_not_merge_when_next_segment_starts_uppercase(self):
        regular_segment_1, beginning_segments, end_segments, regular_segment_2 = self.get_segments()

        end_segments.text_content = "Here"

        segments = [regular_segment_1, beginning_segments, end_segments, regular_segment_2]
        language_segment = SegmentsFromLanguage(language="en", segments=segments, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphExtractor(extractor_identifier=self.extraction_identifier)
        segments_from_languages = [language_segment]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual("Text.", segments_from_languages[0].segments[0].text_content)
        self.assertEqual("Text to be continued", segments_from_languages[0].segments[1].text_content)
        self.assertEqual("Here", segments_from_languages[0].segments[2].text_content)
        self.assertEqual("Text.", segments_from_languages[0].segments[3].text_content)

    def test_not_merge_when_next_segment_starts_with_number(self):
        regular_segment_1, beginning_segments, end_segments, regular_segment_2 = self.get_segments()

        end_segments.text_content = "1. Here"

        segments = [regular_segment_1, beginning_segments, end_segments, regular_segment_2]
        language_segment = SegmentsFromLanguage(language="en", segments=segments, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphExtractor(extractor_identifier=self.extraction_identifier)
        segments_from_languages = [language_segment]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual("Text.", segments_from_languages[0].segments[0].text_content)
        self.assertEqual("Text to be continued", segments_from_languages[0].segments[1].text_content)
        self.assertEqual("1. Here", segments_from_languages[0].segments[2].text_content)
        self.assertEqual("Text.", segments_from_languages[0].segments[3].text_content)

    def test_not_merge_when_next_segment_from_other_type(self):
        regular_segment_1, beginning_segments, end_segments, regular_segment_2 = self.get_segments()

        end_segments.segment_type = TokenType.TITLE

        segments = [regular_segment_1, beginning_segments, end_segments, regular_segment_2]
        language_segment = SegmentsFromLanguage(language="en", segments=segments, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphExtractor(extractor_identifier=self.extraction_identifier)
        segments_from_languages = [language_segment]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual("Text.", segments_from_languages[0].segments[0].text_content)
        self.assertEqual("Text to be continued", segments_from_languages[0].segments[1].text_content)
        self.assertEqual("here", segments_from_languages[0].segments[2].text_content)
        self.assertEqual("Text.", segments_from_languages[0].segments[3].text_content)
