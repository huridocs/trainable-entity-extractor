from pydantic import BaseModel

from trainable_entity_extractor.use_cases.extractors.segment_selector.ParagraphSegmentBox import ParagraphSegmentBox


class Paragraphs(BaseModel):
    page_width: int
    page_height: int
    paragraphs: list[ParagraphSegmentBox]
