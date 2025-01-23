from pydantic import BaseModel
from trainable_entity_extractor.data.PdfDataSegment import PdfDataSegment


class SegmentsFromLanguage(BaseModel):
    language: str
    segments: list[PdfDataSegment]
    is_main_language: bool

    class Config:
        arbitrary_types_allowed = True
