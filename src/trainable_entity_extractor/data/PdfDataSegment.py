from statistics import mode

from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType
from pdf_features.PdfToken import PdfToken


class PdfDataSegment:
    def __init__(
        self, page_number: int, bounding_box: Rectangle, text_content: str, segment_type: TokenType = TokenType.TEXT
    ):
        self.page_number = page_number
        self.bounding_box = bounding_box
        self.text_content = text_content
        self.ml_label = 0
        self.segment_type: TokenType = segment_type

    def is_selected(self, bounding_box: Rectangle):
        if bounding_box.bottom < self.bounding_box.top or self.bounding_box.bottom < bounding_box.top:
            return False

        if bounding_box.right < self.bounding_box.left or self.bounding_box.right < bounding_box.left:
            return False

        return True

    def intersects(self, pdf_segment: "PdfDataSegment"):
        if self.page_number != pdf_segment.page_number:
            return False

        return pdf_segment.bounding_box.get_intersection_percentage(self.bounding_box) > 50

    @staticmethod
    def from_pdf_token(pdf_token: PdfToken):
        return PdfDataSegment(
            page_number=pdf_token.page_number,
            bounding_box=pdf_token.bounding_box,
            text_content=pdf_token.content,
            segment_type=pdf_token.token_type,
        )

    @staticmethod
    def from_pdf_tokens(pdf_tokens: list[PdfToken]):
        text: str = " ".join([pdf_token.content for pdf_token in pdf_tokens])
        bounding_boxes = [pdf_token.bounding_box for pdf_token in pdf_tokens]
        segment_type = mode([token.token_type for token in pdf_tokens])
        return PdfDataSegment(pdf_tokens[0].page_number, Rectangle.merge_rectangles(bounding_boxes), text, segment_type)

    @staticmethod
    def from_list_to_merge(pdf_segments_to_merge: list["PdfDataSegment"]):
        text_content = " ".join([pdf_segment.text_content for pdf_segment in pdf_segments_to_merge])
        bounding_box = Rectangle.merge_rectangles([pdf_segment.bounding_box for pdf_segment in pdf_segments_to_merge])
        segment_type = mode([segment.segment_type for segment in pdf_segments_to_merge])
        return PdfDataSegment(
            page_number=pdf_segments_to_merge[0].page_number,
            bounding_box=bounding_box,
            text_content=text_content,
            segment_type=segment_type,
        )

    @staticmethod
    def from_token_list_to_merge(tokens: list[PdfToken]):
        return PdfDataSegment.from_list_to_merge([PdfDataSegment.from_pdf_token(token) for token in tokens])

    @staticmethod
    def from_texts(texts: list[str]):
        return [PdfDataSegment(i + 1, Rectangle(0, 0, 0, 0), text) for i, text in enumerate(texts)]

    @staticmethod
    def from_text(text: str):
        return PdfDataSegment(0, Rectangle(0, 0, 0, 0), text)

    def are_similar(self, other: "PdfDataSegment"):
        segment_numbers = [x for x in self.text_content if x.isnumeric()]
        other_segment_numbers = [x for x in other.text_content if x.isnumeric()]
        if segment_numbers and segment_numbers == other_segment_numbers:
            return True

        words_number = len(self.text_content.split())
        if words_number < 5 and segment_numbers and other_segment_numbers and segment_numbers != other_segment_numbers:
            return False

        excluded_symbols = [" ", ".", "!", "?", ";", ","]
        symbols = [x for x in self.text_content if not x.isalnum() and x not in excluded_symbols]
        other_symbols = [x for x in other.text_content if not x.isalnum() and x not in excluded_symbols]
        if len(symbols) > 2 and symbols == other_symbols:
            return True

        other_words_number = len(other.text_content.split())
        if words_number < 5 and abs(words_number - other_words_number) < 3:
            return True

        return False
