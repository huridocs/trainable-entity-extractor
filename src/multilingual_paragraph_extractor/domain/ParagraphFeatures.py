from typing import Optional

from pdf_features.PdfFont import PdfFont
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType
from pydantic import BaseModel
from unidecode import unidecode

from trainable_entity_extractor.data.PdfData import PdfData
from trainable_entity_extractor.data.PdfDataSegment import PdfDataSegment


class ParagraphFeatures(BaseModel):
    index: int = 0
    page_height: int = 0
    page_width: int = 0
    paragraph_type: TokenType = TokenType.TEXT
    page_number: int = 1
    bounding_box: Rectangle = Rectangle.from_coordinates(0, 0, 0, 0)
    text_cleaned: str = ""
    original_text: str = ""
    words: list[str] = []
    numbers: list[int] = []
    non_alphanumeric_characters: list[str] = []
    first_word: Optional[str] = None
    font: Optional[PdfFont] = None
    __hash__ = object.__hash__

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def get_empty():
        return ParagraphFeatures()

    @staticmethod
    def from_pdf_data(pdf_data: PdfData, pdf_segment: PdfDataSegment) -> "ParagraphFeatures":
        non_alphanumeric_characters = [x for x in pdf_segment.text_content if not x.isalnum() and x != " "]
        first_token = ParagraphFeatures.get_first_token(pdf_data, pdf_segment)
        words = pdf_segment.text_content.split()
        numbers = ["".join([y for y in x if y.isnumeric() and y.isascii()]) for x in words]
        numbers = [int(x) for x in numbers if x]

        return ParagraphFeatures(
            index=pdf_data.pdf_data_segments.index(pdf_segment),
            page_height=pdf_data.pdf_features.pages[0].page_height if pdf_data.pdf_features.pages else 1,
            page_width=pdf_data.pdf_features.pages[0].page_width if pdf_data.pdf_features.pages else 1,
            page_number=pdf_segment.page_number,
            bounding_box=pdf_segment.bounding_box,
            text_cleaned=" ".join(unidecode(pdf_segment.text_content).split()),
            original_text=" ".join(pdf_segment.text_content.split()),
            paragraph_type=pdf_segment.segment_type,
            words=words,
            numbers=numbers,
            non_alphanumeric_characters=list(non_alphanumeric_characters),
            first_word=unidecode(pdf_segment.text_content.split()[0]) if pdf_segment.text_content else None,
            font=first_token.font if first_token else None,
        )

    @staticmethod
    def get_first_token(pdf_data, pdf_segment):
        for page, token in pdf_data.pdf_features.loop_tokens():
            if token.page_number != pdf_segment.page_number:
                continue

            if pdf_segment.is_selected(token.bounding_box):
                return token

        return None

    @staticmethod
    def from_texts(texts: list[str]):
        paragraphs_features = []
        for text in texts:
            non_alphanumeric_characters = [x for x in text if not x.isalnum() and x != " "]
            words = text.split()
            numbers = ["".join([y for y in x if y.isnumeric()]) for x in words]
            numbers = [int(x) for x in numbers if x]
            paragraphs_features.append(
                ParagraphFeatures(
                    text_cleaned=text,
                    page_width=10,
                    page_height=10,
                    font=PdfFont(font_id="1", bold=False, italics=False, font_size=10, color="#000000"),
                    first_word=text.split()[0],
                    words=text.split(),
                    numbers=numbers,
                    non_alphanumeric_characters=list(non_alphanumeric_characters),
                )
            )
        return paragraphs_features

    def merge(self, paragraph_features: "ParagraphFeatures") -> "ParagraphFeatures":
        self.text_cleaned += " " + paragraph_features.text_cleaned
        self.words += paragraph_features.words
        self.numbers += paragraph_features.numbers
        self.non_alphanumeric_characters += paragraph_features.non_alphanumeric_characters
        return self
