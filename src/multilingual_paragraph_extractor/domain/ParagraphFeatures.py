from typing import Optional

from pdf_features.PdfFont import PdfFont
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType
from pydantic import BaseModel
from unidecode import unidecode

from trainable_entity_extractor.data.PdfData import PdfData
from trainable_entity_extractor.data.PdfDataSegment import PdfDataSegment

TO_AVOID_BEING_MERGED = [
    TokenType.FORMULA,
    TokenType.FOOTNOTE,
    TokenType.TABLE,
    TokenType.PICTURE,
    TokenType.TITLE,
    TokenType.PAGE_HEADER,
    TokenType.SECTION_HEADER,
    TokenType.CAPTION,
    TokenType.PAGE_FOOTER,
]


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
    numbers_by_spaces: list[int] = []
    non_alphanumeric_characters: list[str] = []
    first_word: Optional[str] = None
    font: Optional[PdfFont] = None
    first_token_bounding_box: Optional[Rectangle] = None
    last_token_bounding_box: Optional[Rectangle] = None
    __hash__ = object.__hash__

    def merge(self, paragraph_features: "ParagraphFeatures") -> "ParagraphFeatures":
        self.text_cleaned += " " + paragraph_features.text_cleaned
        self.original_text += " " + paragraph_features.original_text
        self.words += paragraph_features.words
        self.numbers += paragraph_features.numbers
        self.numbers_by_spaces += paragraph_features.numbers_by_spaces
        self.non_alphanumeric_characters += paragraph_features.non_alphanumeric_characters
        return self

    def split_paragraph(self, splitter_word: str) -> ("ParagraphFeatures", "ParagraphFeatures"):
        paragraph_text_1 = self.original_text.split(splitter_word)[0]
        text_cleaned = " ".join(unidecode(paragraph_text_1).split())
        words = paragraph_text_1.split()
        numbers_by_spaces, numbers = ParagraphFeatures.get_numbers(words)
        paragraph_1 = ParagraphFeatures(
            index=self.index,
            page_height=self.page_height,
            page_width=self.page_width,
            paragraph_type=self.paragraph_type,
            page_number=self.page_number,
            bounding_box=self.bounding_box,
            text_cleaned=text_cleaned,
            original_text=paragraph_text_1,
            words=words,
            numbers=numbers,
            numbers_by_spaces=numbers_by_spaces,
            non_alphanumeric_characters=ParagraphFeatures.get_aphanumeric(paragraph_text_1),
            first_word=words[0],
            font=self.font,
            first_token_bounding_box=self.first_token_bounding_box,
            last_token_bounding_box=self.last_token_bounding_box,
        )

        paragraph_text_2 = splitter_word + self.original_text.split(splitter_word)[1]
        text_cleaned = " ".join(unidecode(paragraph_text_2).split())
        words = paragraph_text_2.split()
        numbers_by_spaces, numbers = ParagraphFeatures.get_numbers(words)
        paragraph_2 = ParagraphFeatures(
            index=self.index,
            page_height=self.page_height,
            page_width=self.page_width,
            paragraph_type=self.paragraph_type,
            page_number=self.page_number,
            bounding_box=self.bounding_box,
            text_cleaned=text_cleaned,
            original_text=paragraph_text_2,
            words=words,
            numbers=numbers,
            numbers_by_spaces=numbers_by_spaces,
            non_alphanumeric_characters=ParagraphFeatures.get_aphanumeric(paragraph_text_2),
            first_word=words[0],
            font=self.font,
            first_token_bounding_box=self.first_token_bounding_box,
            last_token_bounding_box=self.last_token_bounding_box,
        )

        return paragraph_1, paragraph_2

    def is_part_of_same_segment(self, next_segment: "ParagraphFeatures") -> bool:
        if self.page_number == next_segment.page_number:
            return False

        if int(self.page_number - next_segment.page_number) > 1:
            return False

        if self.paragraph_type in TO_AVOID_BEING_MERGED or next_segment.paragraph_type in TO_AVOID_BEING_MERGED:
            return False

        if self.text_cleaned[-1] in [".", "!", "?", ";"]:
            return False

        if not next_segment.text_cleaned[0].isalnum():
            return False

        if self.last_token_bounding_box.right < self.bounding_box.right - 0.1 * self.bounding_box.width:
            return False

        if self.last_token_bounding_box.right < self.page_width - 0.2 * self.page_width:
            return False

        return True

    @staticmethod
    def get_empty():
        return ParagraphFeatures()

    @staticmethod
    def get_numbers(words):
        numbers_and_spaces_list = ["".join([y if y.isnumeric() and y.isascii() else " " for y in x]) for x in words]
        numbers_by_spaces = []
        for numbers_and_spaces in numbers_and_spaces_list:
            if not numbers_and_spaces:
                continue
            numbers_by_spaces.extend([int(x) for x in numbers_and_spaces.split() if x])

        merged_number_words = []
        for idx, word in enumerate(words):
            if not merged_number_words:
                merged_number_words.append(word)
                continue

            if not word.isnumeric():
                merged_number_words.append(word)
                continue

            if merged_number_words[-1].isnumeric() and word.isnumeric():
                merged_number_words[-1] += word
                continue

            merged_number_words.append(word)

        numbers = ["".join([y for y in x if y.isnumeric() and y.isascii()]) for x in merged_number_words]
        numbers = [int(x) for x in numbers if x]

        return numbers_by_spaces, numbers

    @staticmethod
    def get_aphanumeric(text):
        return [x for x in text if not x.isalnum() and x != " "]

    @staticmethod
    def from_pdf_data(pdf_data: PdfData, pdf_segment: PdfDataSegment) -> "ParagraphFeatures":
        first_token = ParagraphFeatures.get_first_token(pdf_data, pdf_segment)
        words = pdf_segment.text_content.split()
        numbers_by_spaces, numbers = ParagraphFeatures.get_numbers(words)
        tokens = list()
        for page, token in pdf_data.pdf_features.loop_tokens():
            if token.page_number != pdf_segment.page_number:
                continue

            if pdf_segment.is_selected(token.bounding_box):
                tokens.append(token)

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
            numbers_by_spaces=numbers_by_spaces,
            non_alphanumeric_characters=ParagraphFeatures.get_aphanumeric(pdf_segment.text_content),
            first_word=unidecode(pdf_segment.text_content.split()[0]) if pdf_segment.text_content else None,
            font=first_token.font if first_token else None,
            first_token_bounding_box=tokens[0].bounding_box if tokens else None,
            last_token_bounding_box=tokens[-1].bounding_box if tokens else None,
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
            numbers_by_spaces, numbers = ParagraphFeatures.get_numbers(words)
            paragraphs_features.append(
                ParagraphFeatures(
                    original_text=text,
                    text_cleaned=text,
                    page_width=10,
                    page_height=10,
                    font=PdfFont(font_id="1", bold=False, italics=False, font_size=10, color="#000000"),
                    first_word=text.split()[0],
                    words=text.split(),
                    numbers=numbers,
                    numbers_by_spaces=numbers_by_spaces,
                    non_alphanumeric_characters=list(non_alphanumeric_characters),
                    first_token_bounding_box=Rectangle.from_coordinates(0, 0, 0, 0),
                    last_token_bounding_box=Rectangle.from_coordinates(0, 0, 0, 0),
                )
            )
        return paragraphs_features

    def get_distance(self, next_paragraph: "ParagraphFeatures") -> float:
        if self.page_number != next_paragraph.page_number:
            return 0

        return (next_paragraph.first_token_bounding_box.top - self.last_token_bounding_box.bottom) / self.page_height
