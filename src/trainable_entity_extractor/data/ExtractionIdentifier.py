import os
from os.path import join, exists
from pathlib import Path
from time import time

from pydantic import BaseModel

from trainable_entity_extractor.config import DATA_PATH


class ExtractionIdentifier(BaseModel):
    run_name: str = "default"
    extraction_name: str
    model_path: str | Path = DATA_PATH
    metadata: dict[str, str] = dict()

    def get_path(self):
        return join(self.model_path, self.run_name, self.extraction_name)

    def get_options_path(self):
        path = Path(self.model_path, self.run_name, f"{self.extraction_name}_options.json")
        if not exists(path.parent):
            os.makedirs(path.parent, exist_ok=True)
        return path

    def get_extractor_used_path(self) -> Path:
        path = Path(self.model_path, self.run_name, self.extraction_name, f"{self.extraction_name}.txt")
        if not exists(path.parent):
            os.makedirs(path.parent, exist_ok=True)
        return path

    def is_old(self):
        path = self.get_path()
        return exists(path) and os.path.isdir(path) and os.path.getmtime(path) < (time() - (2 * 24 * 3600))
