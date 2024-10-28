from pydantic import BaseModel

from trainable_entity_extractor.data.Option import Option


class Params(BaseModel):
    id: str
    options: list[Option] = list()
    multi_value: bool = False
    metadata: dict[str, str] = dict()
