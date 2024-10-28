from pydantic import BaseModel

from trainable_entity_extractor.data.Option import Option


class Options(BaseModel):
    tenant: str
    extraction_id: str
    options: list[Option]
