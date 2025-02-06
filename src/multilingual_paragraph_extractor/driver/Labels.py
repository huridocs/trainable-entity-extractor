from pydantic import BaseModel


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
    paragraphs: list[LanguagesTexts]

    def add_paragraph(self, main_text: str, other_text: str):
        self.paragraphs.append(LanguagesTexts(main_language=main_text, other_language=other_text))

    def get_label_file_name(self) -> str:
        base_name = self.main_xml_name.rsplit("_", 1)[0]
        return f"{base_name}_{self.main_language}_{self.other_language}.json"
