from pdf_token_type_labels.TokenType import TokenType
from pydantic import BaseModel


class ParagraphSegmentBox(BaseModel):
    left: float
    top: float
    width: float
    height: float
    page_number: int
    type: TokenType = TokenType.TEXT
