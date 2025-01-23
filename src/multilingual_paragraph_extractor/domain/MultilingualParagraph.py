from pydantic import BaseModel


class MultilingualParagraph(BaseModel):
    languages: list[str]
    texts: list[str]
