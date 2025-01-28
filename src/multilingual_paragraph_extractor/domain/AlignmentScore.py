from pydantic import BaseModel

from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures


class AlignmentScore(BaseModel):
    main_paragraph: ParagraphFeatures
    other_paragraph: ParagraphFeatures
    score: float
