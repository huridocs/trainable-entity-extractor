from unittest import TestCase

from pdf_features.Rectangle import Rectangle

from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures
from multilingual_paragraph_extractor.domain.ParagraphsFromLanguage import ParagraphsFromLanguage
from multilingual_paragraph_extractor.use_cases.MultilingualParagraphAlignerUseCase import (
    MultilingualParagraphAlignerUseCase,
)
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier


class TestMergeSegmentsSpanningTwoPages(TestCase):
    extraction_identifier = ExtractionIdentifier(extraction_name="paragraph_extraction")

    @staticmethod
    def get_paragraphs():
        last_token_bounding_box = Rectangle.from_coordinates(0, 0, 100, 100)
        regular_paragraph_1 = ParagraphFeatures(
            page_number=1,
            original_text="Text.",
            text_cleaned="Text.",
            page_width=100,
            last_token_bounding_box=last_token_bounding_box,
        )
        beginning_paragraphs = ParagraphFeatures(
            page_number=1,
            original_text="Text to be continued",
            text_cleaned="Text to be continued",
            page_width=100,
            last_token_bounding_box=last_token_bounding_box,
        )
        end_paragraphs = ParagraphFeatures(
            page_number=2,
            original_text="here",
            text_cleaned="here",
            page_width=100,
            last_token_bounding_box=last_token_bounding_box,
        )
        regular_paragraph_2 = ParagraphFeatures(
            page_number=2,
            original_text="Text.",
            text_cleaned="Text.",
            page_width=100,
            last_token_bounding_box=last_token_bounding_box,
        )

        return regular_paragraph_1, beginning_paragraphs, end_paragraphs, regular_paragraph_2

    def test_merge_paragraphs_spanning_two_pages(self):
        regular_paragraph_1, beginning_paragraphs, end_paragraphs, regular_paragraph_2 = self.get_paragraphs()

        paragraphs = [regular_paragraph_1, beginning_paragraphs, end_paragraphs, regular_paragraph_2]
        language_paragraph = ParagraphsFromLanguage(language="en", paragraphs=paragraphs, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        paragraphs_from_languages = [language_paragraph]
        multilingual_paragraph_extractor.align_languages(paragraphs_from_languages)

        self.assertEqual(3, len(paragraphs_from_languages[0]._aligned_paragraphs))
        self.assertEqual("Text.", paragraphs_from_languages[0]._aligned_paragraphs[0].text_cleaned)
        self.assertEqual("Text to be continued here", paragraphs_from_languages[0]._aligned_paragraphs[1].text_cleaned)
        self.assertEqual("Text.", paragraphs_from_languages[0]._aligned_paragraphs[2].text_cleaned)

    def test_not_merge_when_same_page(self):
        regular_paragraph_1, beginning_paragraphs, end_paragraphs, regular_paragraph_2 = self.get_paragraphs()

        regular_paragraph_1.page_number = 5
        beginning_paragraphs.page_number = 5
        end_paragraphs.page_number = 5
        regular_paragraph_2.page_number = 5

        paragraphs = [regular_paragraph_1, beginning_paragraphs, end_paragraphs, regular_paragraph_2]
        language_paragraph = ParagraphsFromLanguage(language="en", paragraphs=paragraphs, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        paragraphs_from_languages = [language_paragraph]
        multilingual_paragraph_extractor.align_languages(paragraphs_from_languages)

        self.assertEqual(4, len(paragraphs_from_languages[0]._aligned_paragraphs))
        self.assertEqual("Text.", paragraphs_from_languages[0]._aligned_paragraphs[0].text_cleaned)
        self.assertEqual("Text to be continued", paragraphs_from_languages[0]._aligned_paragraphs[1].text_cleaned)
        self.assertEqual("here", paragraphs_from_languages[0]._aligned_paragraphs[2].text_cleaned)
        self.assertEqual("Text.", paragraphs_from_languages[0]._aligned_paragraphs[3].text_cleaned)

    def test_not_merge_when_ends_with_dot(self):
        regular_paragraph_1, beginning_paragraphs, end_paragraphs, regular_paragraph_2 = self.get_paragraphs()

        beginning_paragraphs.text_cleaned = "Text not to be continued."

        paragraphs = [regular_paragraph_1, beginning_paragraphs, end_paragraphs, regular_paragraph_2]
        language_paragraph = ParagraphsFromLanguage(language="en", paragraphs=paragraphs, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        paragraphs_from_languages = [language_paragraph]
        multilingual_paragraph_extractor.align_languages(paragraphs_from_languages)

        self.assertEqual(4, len(paragraphs_from_languages[0]._aligned_paragraphs))
        self.assertEqual("Text.", paragraphs_from_languages[0]._aligned_paragraphs[0].text_cleaned)
        self.assertEqual("Text not to be continued.", paragraphs_from_languages[0]._aligned_paragraphs[1].text_cleaned)
        self.assertEqual("here", paragraphs_from_languages[0]._aligned_paragraphs[2].text_cleaned)
        self.assertEqual("Text.", paragraphs_from_languages[0]._aligned_paragraphs[3].text_cleaned)

    def test_do_not_merge_when_last_line_finish_in_the_middle(self):
        regular_paragraph_1, beginning_paragraphs, end_paragraphs, regular_paragraph_2 = self.get_paragraphs()
        beginning_paragraphs.last_token_bounding_box = Rectangle.from_coordinates(0, 0, 50, 100)
        paragraphs = [regular_paragraph_1, beginning_paragraphs, end_paragraphs, regular_paragraph_2]
        language_paragraph = ParagraphsFromLanguage(language="en", paragraphs=paragraphs, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        paragraphs_from_languages = [language_paragraph]
        multilingual_paragraph_extractor.align_languages(paragraphs_from_languages)

        self.assertEqual(4, len(paragraphs_from_languages[0]._aligned_paragraphs))
