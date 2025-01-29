from unittest import TestCase

from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType

from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures
from multilingual_paragraph_extractor.domain.SegmentsFromLanguage import ParagraphsFromLanguage
from multilingual_paragraph_extractor.use_cases.MultilingualParagraphAlignerUseCase import (
    MultilingualParagraphAlignerUseCase,
)
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier


class TestMergeSegmentsSpanningTwoPages(TestCase):
    extraction_identifier = ExtractionIdentifier(extraction_name="paragraph_extraction")

    @staticmethod
    def get_segments():
        regular_segment_1 = ParagraphFeatures(page_number=1, text_content="Text.")
        beginning_segments = ParagraphFeatures(page_number=1, text_content="Text to be continued")
        end_segments = ParagraphFeatures(page_number=2, text_content="here")
        regular_segment_2 = ParagraphFeatures(page_number=2, text_content="Text.")

        return regular_segment_1, beginning_segments, end_segments, regular_segment_2

    def test_merge_segments_spanning_two_pages(self):
        regular_segment_1, beginning_segments, end_segments, regular_segment_2 = self.get_segments()

        segments = [regular_segment_1, beginning_segments, end_segments, regular_segment_2]
        language_segment = ParagraphsFromLanguage(language="en", segments=segments, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        segments_from_languages = [language_segment]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(3, len(segments_from_languages[0].paragraphs))
        self.assertEqual("Text.", segments_from_languages[0].paragraphs[0].text_content)
        self.assertEqual("Text to be continued here", segments_from_languages[0].paragraphs[1].text_content)
        self.assertEqual("Text.", segments_from_languages[0].paragraphs[2].text_content)

    def test_not_merge_when_same_page(self):
        regular_segment_1, beginning_segments, end_segments, regular_segment_2 = self.get_segments()

        regular_segment_1.page_number = 5
        beginning_segments.page_number = 5
        end_segments.page_number = 5
        regular_segment_2.page_number = 5

        segments = [regular_segment_1, beginning_segments, end_segments, regular_segment_2]
        language_segment = ParagraphsFromLanguage(language="en", segments=segments, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        segments_from_languages = [language_segment]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(4, len(segments_from_languages[0].paragraphs))
        self.assertEqual("Text.", segments_from_languages[0].paragraphs[0].text_content)
        self.assertEqual("Text to be continued", segments_from_languages[0].paragraphs[1].text_content)
        self.assertEqual("here", segments_from_languages[0].paragraphs[2].text_content)
        self.assertEqual("Text.", segments_from_languages[0].paragraphs[3].text_content)

    def test_not_merge_when_ends_with_dot(self):
        regular_segment_1, beginning_segments, end_segments, regular_segment_2 = self.get_segments()

        beginning_segments.text_content = "Text not to be continued."

        segments = [regular_segment_1, beginning_segments, end_segments, regular_segment_2]
        language_segment = ParagraphsFromLanguage(language="en", segments=segments, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        segments_from_languages = [language_segment]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(4, len(segments_from_languages[0].paragraphs))
        self.assertEqual("Text.", segments_from_languages[0].paragraphs[0].text_content)
        self.assertEqual("Text not to be continued.", segments_from_languages[0].paragraphs[1].text_content)
        self.assertEqual("here", segments_from_languages[0].paragraphs[2].text_content)
        self.assertEqual("Text.", segments_from_languages[0].paragraphs[3].text_content)

    def test_not_merge_when_next_segment_from_other_type(self):
        regular_segment_1, beginning_segments, end_segments, regular_segment_2 = self.get_segments()

        end_segments.segment_type = TokenType.TITLE

        segments = [regular_segment_1, beginning_segments, end_segments, regular_segment_2]
        language_segment = ParagraphsFromLanguage(language="en", segments=segments, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        segments_from_languages = [language_segment]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual("Text.", segments_from_languages[0].paragraphs[0].text_content)
        self.assertEqual("Text to be continued", segments_from_languages[0].paragraphs[1].text_content)
        self.assertEqual("here", segments_from_languages[0].paragraphs[2].text_content)
        self.assertEqual("Text.", segments_from_languages[0].paragraphs[3].text_content)
