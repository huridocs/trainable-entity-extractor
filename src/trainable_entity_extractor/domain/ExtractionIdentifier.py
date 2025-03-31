import json
import os
from os.path import join, exists
from pathlib import Path
from time import time
from typing import Any

from pydantic import BaseModel

from trainable_entity_extractor.config import DATA_PATH
from trainable_entity_extractor.domain.ExtractionStatus import ExtractionStatus
from trainable_entity_extractor.domain.Option import Option

OPTIONS_FILE_NAME = "options.json"
MULTI_VALUE_FILE_NAME = "multi_value.json"
METHOD_USED_FILE_NAME = "method_used.json"
PROCESSING_FINISHED_FILE_NAME = "processing_finished.json"
EXTRACTOR_USED_FILE_NAME = "extractor_used.json"


class ExtractionIdentifier(BaseModel):
    run_name: str = "default"
    output_path: str | Path = DATA_PATH
    extraction_name: str
    metadata: dict[str, str] = dict()

    def get_path(self):
        return join(self.output_path, self.run_name, self.extraction_name)

    def get_file_content(self, file_name: str, default: Any = None) -> Any:
        path = Path(self.get_path(), file_name)
        if not path.exists():
            return default

        return json.loads(path.read_text())

    def save_content(self, file_name: str, content: Any):
        path = Path(self.get_path(), file_name)

        if not exists(path.parent):
            os.makedirs(path.parent, exist_ok=True)

        if type(content) == str:
            path.write_text(content)
        elif type(content) == list:
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

    def get_status(self) -> ExtractionStatus:
        method_used = self.get_method_used()
        if not method_used:
            return ExtractionStatus.NO_MODEL

        if self.get_file_content(PROCESSING_FINISHED_FILE_NAME, False):
            return ExtractionStatus.READY

        return ExtractionStatus.PROCESSING

    def set_extractor_to_processing(self):
        Path(self.get_path(), PROCESSING_FINISHED_FILE_NAME).unlink(missing_ok=True)

    def save_processing_finished(self, success: bool):
        self.save_content(PROCESSING_FINISHED_FILE_NAME, success)
