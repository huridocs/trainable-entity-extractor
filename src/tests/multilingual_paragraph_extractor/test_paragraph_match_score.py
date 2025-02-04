from unittest import TestCase


from pdf_features.PdfFont import PdfFont
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType

from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures
from multilingual_paragraph_extractor.domain.ParagraphMatchScore import ParagraphMatchScore


class TestParagraphMatchScore(TestCase):
    @staticmethod
    def get_paragraph():
        return ParagraphFeatures(
            text_cleaned="text 1",
            page_width=10,
            page_height=10,
            paragraph_type=TokenType.PAGE_HEADER,
            page_number=1,
            bounding_box=Rectangle.from_coordinates(0, 2, 0, 2),
            index=1,
            words=["text"],
            numbers=[1],
            non_alphanumeric_characters=[";", ".", ",", ":"],
            first_word="text",
            font=PdfFont(font_id="1", font_size=10, bold=False, italics=False, color="black"),
        )

    def test_paragraph_match_score(self):
        paragraph = self.get_paragraph()

        match_score = ParagraphMatchScore.from_paragraphs_features(paragraph, paragraph)

        self.assertEqual(1.0, match_score.index)
        self.assertEqual(1.0, match_score.segment_type)
        self.assertEqual(1.0, match_score.page)
        self.assertEqual(1.0, match_score.text_fuzzy_match)
        self.assertEqual(1.0, match_score.number_of_words)
        self.assertEqual(1.0, match_score.numbers)
        self.assertEqual(1.0, match_score.first_word)
        self.assertEqual(1.0, match_score.special_characters)
        self.assertEqual(1.0, match_score.bounding_boxes)
        self.assertEqual(1.0, match_score.alignment)
        self.assertEqual(1.0, match_score.indentation)
        self.assertEqual(1.0, match_score.font_style)
        self.assertEqual(1.0, match_score.font_size)
        self.assertEqual(1.0, match_score.overall_score)

    def test_paragraph_non_matching(self):
        paragraph = self.get_paragraph()

        other_paragraph = ParagraphFeatures(
            text_cleaned="",
            page_width=10,
            page_height=10,
            paragraph_type=TokenType.TEXT,
            page_number=2,
            bounding_box=Rectangle.from_coordinates(10, 2, 10, 4),
            index=3,
            words=[],
            numbers=[],
            non_alphanumeric_characters=[],
            first_word="foo",
            font=PdfFont(font_id="1", font_size=50, bold=True, italics=True, color="black"),
        )

        match_score = ParagraphMatchScore.from_paragraphs_features(paragraph, other_paragraph)

        self.assertEqual(0.0, match_score.index)
        self.assertEqual(0.0, match_score.segment_type)
        self.assertEqual(0.0, match_score.page)
        self.assertEqual(0.0, match_score.text_fuzzy_match)
        self.assertEqual(0.0, match_score.number_of_words)
        self.assertEqual(0.0, match_score.numbers)
        self.assertEqual(0.0, match_score.first_word)
        self.assertEqual(0.0, match_score.special_characters)
        self.assertEqual(0.0, match_score.bounding_boxes)
        self.assertTrue(match_score.alignment < 0.2)
        self.assertEqual(0.0, match_score.indentation)
        self.assertEqual(0.0, match_score.font_style)
        self.assertTrue(match_score.font_size < 0.2)
        self.assertTrue(match_score.overall_score < 0.05)

    def test_paragraph_half_matching(self):
        paragraph = self.get_paragraph()

        other_paragraph = ParagraphFeatures(
            index=2,
            page_number=1,
            text_cleaned="",
            page_width=10,
            page_height=10,
            paragraph_type=TokenType.TEXT,
            bounding_box=Rectangle.from_coordinates(5, 7, 5, 7),
            words=["text", "text"],
            numbers=[1, 2],
            non_alphanumeric_characters=[".", ","],
            first_word="tezz",
            font=PdfFont(font_id="1", font_size=5, bold=True, italics=False, color="black"),
        )

        match_score = ParagraphMatchScore.from_paragraphs_features(paragraph, other_paragraph)

        self.assertEqual(0, match_score.index)
        self.assertEqual(1, match_score.page)
        self.assertEqual(0, match_score.segment_type)
        self.assertEqual(0.5, match_score.text_fuzzy_match)
        self.assertEqual(0.5, match_score.number_of_words)
        self.assertEqual(0.5, match_score.numbers)
        self.assertEqual(0.5, match_score.first_word)
        self.assertEqual(0.5, match_score.special_characters)
        self.assertEqual(0.5, match_score.bounding_boxes)
        self.assertEqual(0.5, match_score.alignment)
        self.assertEqual(0.5, match_score.indentation)
        self.assertEqual(0.5, match_score.font_style)
        self.assertEqual(0.5, match_score.font_size)
        self.assertTrue(0.4 < match_score.overall_score < 0.6)
