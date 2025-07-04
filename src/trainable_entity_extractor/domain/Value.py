from pydantic import BaseModel


class Value(BaseModel):
    id: str
    label: str
    segment_text: str = ""
    __hash__ = object.__hash__
