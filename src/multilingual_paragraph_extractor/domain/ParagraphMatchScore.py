import rapidfuzz
from pydantic import BaseModel
from typing import Optional

from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures


import rapidfuzz
from pydantic import BaseModel
from typing import Optional

from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures


class ParagraphMatchScore(BaseModel):
    index: Optional[float] = None
    segment_type: Optional[float] = None
    page: Optional[float] = None
    text_fuzzy_match: Optional[float] = None
    number_of_words: Optional[float] = None
    numbers: Optional[float] = None
    first_word: Optional[float] = None
    special_characters: Optional[float] = None
    bounding_boxes: Optional[float] = None
    alignment: Optional[float] = None
    indentation: Optional[float] = None
    font_style: Optional[float] = None
    font_size: Optional[float] = None
    segments_around: Optional[float] = None
    overall_score: Optional[float] = None

    @staticmethod
    def from_paragraphs_features(paragraph_1: ParagraphFeatures, paragraph_2: ParagraphFeatures) -> "ParagraphMatchScore":
        return ParagraphMatchScore(
            index=ParagraphMatchScore.are_the_same(paragraph_1.index, paragraph_2.index),
            page=ParagraphMatchScore.are_the_same(paragraph_1.page_number, paragraph_2.page_number),
            segment_type=ParagraphMatchScore.get_segment_type_score(paragraph_1, paragraph_2),
            text_fuzzy_match=ParagraphMatchScore.get_text_fuzzy_match_score(paragraph_1, paragraph_2),
            number_of_words=ParagraphMatchScore.get_number_of_words_score(paragraph_1, paragraph_2),
            numbers=ParagraphMatchScore.get_numbers_score(paragraph_1, paragraph_2),
            first_word=ParagraphMatchScore.get_first_word_score(paragraph_1, paragraph_2),
            special_characters=ParagraphMatchScore.get_special_characters_score(paragraph_1, paragraph_2),
            bounding_boxes=ParagraphMatchScore.get_bounding_boxes_score(paragraph_1, paragraph_2),
            alignment=ParagraphMatchScore.get_alignment_score(paragraph_1, paragraph_2),
            indentation=ParagraphMatchScore.get_indentation_score(paragraph_1, paragraph_2),
            font_size=ParagraphMatchScore.get_font_size_score(paragraph_1, paragraph_2),
            font_style=ParagraphMatchScore.get_font_style_score(paragraph_1, paragraph_2),
            overall_score=None,
        )

    @staticmethod
    def are_the_same(value_1: int, value_2: int) -> float:
        return 1.0 if value_1 == value_2 else 0.0

    @staticmethod
    def get_segment_type_score(paragraph_1: ParagraphFeatures, paragraph_2: ParagraphFeatures) -> float:
        return 1.0 if paragraph_1.segment_type == paragraph_2.segment_type else 0.0

    @staticmethod
    def get_text_fuzzy_match_score(paragraph_1: ParagraphFeatures, paragraph_2: ParagraphFeatures) -> float:
        matching_words = len([x for x in paragraph_1.words if x in paragraph_2.words])
        max_words = max(len(paragraph_1.words), len(paragraph_2.words))
        return matching_words / max_words if paragraph_1.words else 0

    @staticmethod
    def get_page_score(paragraph_1: ParagraphFeatures, paragraph_2: ParagraphFeatures) -> float:
        return 1.0 if paragraph_1.page_number == paragraph_2.page_number else 0.0

    @staticmethod
    def get_number_of_words_score(paragraph_1: ParagraphFeatures, paragraph_2: ParagraphFeatures) -> float:
        words_difference = abs(len(paragraph_1.words) - len(paragraph_2.words))
        max_words = max(len(paragraph_1.words), len(paragraph_2.words))
        return (1 - words_difference / max_words) if paragraph_1.words else 0

    @staticmethod
    def get_numbers_score(paragraph_1: ParagraphFeatures, paragraph_2: ParagraphFeatures) -> float:
        max_numbers = max(len(paragraph_1.numbers), len(paragraph_2.numbers))
        return len([x for x in paragraph_1.numbers if x in paragraph_2.numbers]) / max_numbers

    @staticmethod
    def get_first_word_score(paragraph_1: ParagraphFeatures, paragraph_2: ParagraphFeatures) -> float:
        return rapidfuzz.fuzz.ratio(paragraph_1.first_word, paragraph_2.first_word) / 100

    @staticmethod
    def get_special_characters_score(paragraph_1: ParagraphFeatures, paragraph_2: ParagraphFeatures) -> float:
        if paragraph_1.non_alphanumeric_characters:
            same_special_characters = len(
                [x for x in paragraph_1.non_alphanumeric_characters if x in paragraph_2.non_alphanumeric_characters]
            )
            return same_special_characters / len(paragraph_1.non_alphanumeric_characters)
        elif paragraph_2.non_alphanumeric_characters:
            return 0
        else:
            return 1

    @staticmethod
    def get_bounding_boxes_score(paragraph_1: ParagraphFeatures, paragraph_2: ParagraphFeatures) -> float:
        if paragraph_1.page_number != paragraph_2.page_number:
            return 0
        vertical_center_1 = paragraph_1.bounding_box.top + paragraph_1.bounding_box.height / 2
        vertical_center_2 = paragraph_2.bounding_box.top + paragraph_2.bounding_box.height / 2
        vertical_distance = abs(vertical_center_1 - vertical_center_2)
        return 1 - vertical_distance / paragraph_1.page_height if paragraph_1.page_height else 0

    @staticmethod
    def get_alignment_score(paragraph_1: ParagraphFeatures, paragraph_2: ParagraphFeatures) -> float:
        right_margin_1 = abs(paragraph_1.page_width - paragraph_1.bounding_box.right)
        right_margin_2 = abs(paragraph_2.page_width - paragraph_2.bounding_box.right)
        margins_difference = abs(right_margin_1 - right_margin_2)
        return 1 - margins_difference / paragraph_1.page_width if paragraph_1.page_width else 0

    @staticmethod
    def get_indentation_score(paragraph_1: ParagraphFeatures, paragraph_2: ParagraphFeatures) -> float:
        horizontal_center_1 = paragraph_1.bounding_box.left + paragraph_1.bounding_box.width / 2
        horizontal_center_2 = paragraph_2.bounding_box.left + paragraph_2.bounding_box.width / 2
        horizontal_distance = abs(horizontal_center_1 - horizontal_center_2)
        return (1 - horizontal_distance / paragraph_1.page_width) if paragraph_1.page_width else 0

    @staticmethod
    def get_font_size_score(paragraph_1: ParagraphFeatures, paragraph_2: ParagraphFeatures) -> float:
        font_size_difference = abs(paragraph_1.font.font_size - paragraph_2.font.font_size)
        max_font_size = max(paragraph_1.font.font_size, paragraph_2.font.font_size)
        return 1 - font_size_difference / max_font_size if paragraph_1.font.font_size else 0

    @staticmethod
    def get_font_style_score(paragraph_1: ParagraphFeatures, paragraph_2: ParagraphFeatures) -> float:
        font_style = 0.5 if paragraph_1.font.bold == paragraph_2.font.bold else 0
        font_style += 0.5 if paragraph_1.font.italics == paragraph_2.font.italics else 0
        return font_style
