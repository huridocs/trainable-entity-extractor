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
            numbers_by_spaces=[1],
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
            numbers_by_spaces=[],
            non_alphanumeric_characters=[],
            first_word="foo",
            font=PdfFont(font_id="1", font_size=50, bold=True, italics=True, color="black"),
        )

        match_score = ParagraphMatchScore.from_paragraphs_features(paragraph, other_paragraph)

        self.assertEqual(0.5, match_score.index)
        self.assertEqual(0.0, match_score.segment_type)
        self.assertEqual(1, match_score.page)
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
            numbers_by_spaces=[1, 2],
            numbers=[1, 2],
            non_alphanumeric_characters=[".", ","],
            first_word="tezz",
            font=PdfFont(font_id="1", font_size=5, bold=True, italics=False, color="black"),
        )

        match_score = ParagraphMatchScore.from_paragraphs_features(paragraph, other_paragraph)

        self.assertEqual(1, match_score.index)
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

    def test_match_example(self):
        p1, p2 = ParagraphFeatures.from_texts(
            [
                """118.1
Ratify the remaining core international human rights treaties
(Ukraine);""",
                """118.1
Ratifier ceux des principaux instruments internationaux relatifs aux
droits de l’homme auxquels le pays n’est pas encore partie (Ukraine) ;""",
            ]
        )

        p3, p4 = ParagraphFeatures.from_texts(
            [
                """118.11
Consider the ratification of the Optional Protocol to the Convention on
the Elimination of All Forms of Discrimination against Women (Chile);""",
                """118.1
Ratifier ceux des principaux instruments internationaux relatifs aux
droits de l’homme auxquels le pays n’est pas encore partie (Ukraine) ;""",
            ]
        )

        score_1 = ParagraphMatchScore.from_paragraphs_features(p1, p2).overall_score
        score_2 = ParagraphMatchScore.from_paragraphs_features(p3, p4).overall_score
        self.assertGreater(score_1, score_2)

    def test_match_example_2(self):
        p1, p2 = ParagraphFeatures.from_texts(
            [
                """88.1
Ratify the Optional Protocol to the International Covenant on Civil
and Political Rights and the Second Optional Protocol to the International
Covenant on Civil and Political Rights, aiming at the abolition of the death
penalty (Cyprus);""",
                """88.1
Ratifier le Protocole facultatif se rapportant au Pacte international
relatif aux droits civils et politiques et le deuxième Protocole facultatif se
rapportant au Pacte international relatif aux droits civils et politiques, visant à
abolir la peine de mort (Chypre) ;""",
            ]
        )

        p3, p4 = ParagraphFeatures.from_texts(
            [
                """88.1
Ratify the Optional Protocol to the International Covenant on Civil
and Political Rights and the Second Optional Protocol to the International
Covenant on Civil and Political Rights, aiming at the abolition of the death
penalty (Cyprus);""",
                """88.11
Instaurer un moratoire de jure sur la peine de mort et ratifier le
deuxième Protocole facultatif se rapportant au Pacte international relatif aux
droits civils et politiques, visant à abolir la peine de mort (Italie) ;""",
            ]
        )

        score_1 = ParagraphMatchScore.from_paragraphs_features(p1, p2).overall_score
        score_2 = ParagraphMatchScore.from_paragraphs_features(p3, p4).overall_score
        print(score_1, score_2)
        self.assertGreater(score_1, score_2)
