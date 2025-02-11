from pydantic import BaseModel


class AlignmentResult(BaseModel):
    name: str
    algorithm: str
    precision: float
    recall: float
    f1_score: float
    mistakes_number: int
    total_paragraphs: int
    seconds: float
