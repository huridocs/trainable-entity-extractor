from unittest import TestCase

from pdf_features.PdfFont import PdfFont
from pdf_features.PdfToken import PdfToken
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType
from pdf_features.PdfTokenStyle import PdfTokenStyle

from trainable_entity_extractor.domain.PdfData import PdfData


class TestPDFData(TestCase):
    @staticmethod
    def create_token(content: str, font_size: int, left: int = 0):
        font_12 = PdfFont(font_id="1", font_size=font_size, bold=False, italics=False, color="black")
        bounding_box = Rectangle.from_width_height(left, 0, 0, 0)
        token_style = PdfTokenStyle(font=font_12)
        token = PdfToken(
            page_number=1,
            id="tag",
            content=content,
            font=font_12,
            reading_order_no=0,
            bounding_box=bounding_box,
            token_type=TokenType.TEXT,
            token_style=token_style,
        )
        return token

    def test_no_remove_super_scripts(self):
        token_1 = self.create_token("bu", 12, left=1)
        token_2 = self.create_token("1", 12, left=2)
        token_3 = self.create_token("2", 12, left=3)
        tokens = PdfData.remove_super_scripts([token_1, token_2, token_3])

        self.assertEqual(3, len(tokens))

    def test_remove_super_scripts(self):
        token_1 = self.create_token("first", 12)
        token_2 = self.create_token("1", 10, left=1)

        tokens = PdfData.remove_super_scripts([token_1, token_2])

        self.assertEqual(1, len(tokens))
        self.assertEqual("first", tokens[0].content)

    def test_no_remove_super_scripts_when_bigger(self):
        token_1 = self.create_token("foo", 12)
        token_2 = self.create_token("1", 12, left=1)
        token_3 = self.create_token("first", 10, left=2)

        tokens = PdfData.remove_super_scripts([token_1, token_2, token_3])

        self.assertEqual(3, len(tokens))
