from pydantic import BaseModel

from multilingual_paragraph_extractor.domain.AlignmentScore import AlignmentScore


class LanguagesTexts(BaseModel):
    main_language: str
    other_language: str

    def __eq__(self, other: "LanguagesTexts"):
        return self.main_language == other.main_language and self.other_language == other.other_language


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
