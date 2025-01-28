from pydantic import BaseModel

from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures


class SegmentsFromLanguage(BaseModel):
    language: str
    segments: list[ParagraphFeatures]
    is_main_language: bool

    class Config:
        arbitrary_types_allowed = True
