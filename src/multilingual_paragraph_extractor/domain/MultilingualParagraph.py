from pydantic import BaseModel

from trainable_entity_extractor.data.PdfDataSegment import PdfDataSegment


class MultilingualParagraph(BaseModel):
    languages: list[str]
    segments: list[PdfDataSegment]
