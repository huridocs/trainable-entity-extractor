from difflib import SequenceMatcher

from pydantic import BaseModel
from multilingual_paragraph_extractor.domain.AlignmentScore import AlignmentScore


class LanguagesTexts(BaseModel):
    main_language: str
    other_language: str

    @staticmethod
    def get_bidirectional_differences(text_1: str, text_2) -> str:
        matcher = SequenceMatcher(None, text_1, text_2)
        differences = {"text_1_unique": [], "text_2_unique": []}

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace":
                differences["text_1_unique"].append(text_1[i1:i2])
                differences["text_2_unique"].append(text_2[j1:j2])
            elif tag == "delete":
                differences["text_1_unique"].append(text_1[i1:i2])
            elif tag == "insert":
                differences["text_2_unique"].append(text_2[j1:j2])

        return "".join(differences["text_1_unique"] + differences["text_2_unique"])

    def are_similar_texts(self, text_1: str, text_2: str) -> bool:
        if abs(len(text_1) - len(text_2)) > 4:
            return False

        differences = self.get_bidirectional_differences(text_1, text_2)
        are_only_numbers = all([x.isnumeric() for x in differences])

        if not are_only_numbers:
            return False

        if len(differences) > 4:
            return False

        if len(differences) > len(text_1) * 0.05:
            return False

        if len(differences) > len(text_2) * 0.05:
            return False

        return True

    def __eq__(self, other: "LanguagesTexts"):
        are_main_same = self.main_language == other.main_language
        are_other_same = self.other_language == other.other_language

        if are_main_same and are_other_same:
            return True

        are_main_similar = self.are_similar_texts(self.main_language, other.main_language)
        are_other_similar = self.are_similar_texts(self.other_language, other.other_language)

        if are_main_similar and are_other_similar:
            return True

        return False


class Labels(BaseModel):
    main_language: str
    other_language: str
    main_xml_name: str
    other_xml_name: str
    paragraphs: list[LanguagesTexts] = []
    _seconds: float = 0
    _alignment_scores: list[AlignmentScore] = []

    def get_seconds(self) -> float:
        return self._seconds

    def add_seconds(self, seconds: float):
        self._seconds = seconds

    def get_alignment_scores(self) -> list[AlignmentScore]:
        return self._alignment_scores

    def add_paragraph(self, alignment_score: AlignmentScore):
        texts = LanguagesTexts(
            main_language=alignment_score.main_paragraph.original_text,
            other_language=alignment_score.other_paragraph.original_text,
        )
        self.paragraphs.append(texts)
        self._alignment_scores.append(alignment_score)

    def get_label_file_name(self) -> str:
        base_name = self.main_xml_name.rsplit("_", 1)[0]
        return f"{base_name}_{self.main_language}_{self.other_language}.json"

    def get_main_pdf_name(self) -> str:
        return self.main_xml_name.replace(".xml", ".pdf")

    def get_mistakes_pdf_name(self) -> str:
        base_name = self.main_xml_name.rsplit("_", 1)[0]
        return f"{base_name}_{self.main_language}_{self.other_language}_mistakes.pdf"
