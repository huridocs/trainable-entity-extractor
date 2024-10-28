import json
import os
from os.path import join, exists
from pathlib import Path
from time import time
from typing import Any

from pydantic import BaseModel

from trainable_entity_extractor.config import DATA_PATH
from trainable_entity_extractor.data.Option import Option

OPTIONS_FILE_NAME = "options.json"
MULTI_VALUE_FILE_NAME = "multi_value.json"
METHOD_USED_FILE_NAME = "method_used.json"
EXTRACTOR_USED_FILE_NAME = "extractor_used.json"


class ExtractionIdentifier(BaseModel):
    run_name: str = "default"
    extraction_name: str
    model_path: str | Path = DATA_PATH
    metadata: dict[str, str] = dict()

    def get_path(self):
        return join(self.model_path, self.run_name, self.extraction_name)

    def get_file_content(self, file_name: str, default: Any = None) -> Any:
        path = Path(self.get_path(), file_name)
        if not path.exists():
            return default

        return json.loads(path.read_text())

    def save_content(self, file_name: str, content: Any):
        path = Path(self.get_path(), file_name)

        if not exists(path.parent):
            os.makedirs(path.parent, exist_ok=True)

        if type(content) == list:
            path.write_text(json.dumps([x.model_dump() for x in content]))
        else:
            path.write_text(json.dumps(content))

    def get_options_path(self):
        return Path(self.get_path(), OPTIONS_FILE_NAME)

    def get_options(self) -> list[Option]:
        options_dict = self.get_file_content(OPTIONS_FILE_NAME, [])
        return [Option(**x) for x in options_dict]

    def save_options(self, options: list[Option]):
        self.save_content(OPTIONS_FILE_NAME, options)

    def get_multi_value(self) -> bool:
        return self.get_file_content(MULTI_VALUE_FILE_NAME, False)

    def save_multi_value(self, multi_value: bool):
        self.save_content(MULTI_VALUE_FILE_NAME, multi_value)

    def get_method_used(self) -> str:
        return self.get_file_content(METHOD_USED_FILE_NAME, "")

    def save_method_used(self, method_used: str):
        self.save_content(METHOD_USED_FILE_NAME, method_used)

    def get_extractor_used(self) -> str:
        return self.get_file_content(EXTRACTOR_USED_FILE_NAME, "")

    def save_extractor_used(self, method_used: str):
        self.save_content(EXTRACTOR_USED_FILE_NAME, method_used)

    def is_old(self):
        path = self.get_path()
        return exists(path) and os.path.isdir(path) and os.path.getmtime(path) < (time() - (2 * 24 * 3600))
