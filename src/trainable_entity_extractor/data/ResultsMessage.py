from typing import Optional

from pydantic import BaseModel
from trainable_entity_extractor.data.Params import Params


class ResultsMessage(BaseModel):
    tenant: str
    task: str
    params: Params
    success: bool
    error_message: str
    data_url: Optional[str] = None

    def to_string(self):
        return f"tenant: {self.tenant}, id: {self.params.id}, task: {self.task}, success: {self.success}, error_message: {self.error_message}"
