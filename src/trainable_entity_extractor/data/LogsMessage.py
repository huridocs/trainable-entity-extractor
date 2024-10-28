import json
from enum import Enum

from pydantic import BaseModel


class Severity(str, Enum):
    error = "error"
    info = "info"


class LogsMessage(BaseModel):
    tenant: str
    extraction_name: str
    severity: Severity
    message: str

    def dump(self):
        return json.loads(self.model_dump_json())
