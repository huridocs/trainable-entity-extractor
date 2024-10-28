from pydantic import BaseModel
from trainable_entity_extractor.data.Params import Params


class ExtractionTask(BaseModel):
    tenant: str
    task: str
    params: Params
