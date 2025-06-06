from pydantic import BaseModel


class Option(BaseModel):
    id: str
    label: str
    __hash__ = object.__hash__
