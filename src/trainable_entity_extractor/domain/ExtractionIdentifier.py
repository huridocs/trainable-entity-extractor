import json
import os
import shutil
from os.path import join, exists
from pathlib import Path
from time import time
from typing import Any

from pydantic import BaseModel

from trainable_entity_extractor.config import DATA_PATH

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
    extra_model_folder: str = ""

    def set_extra_model_folder(self, folder: str):
        extraction_identifier = self.model_copy()
        extraction_identifier.extra_model_folder = folder
        return extraction_identifier

    def get_path(self):
        if self.extra_model_folder == "":
            return join(self.output_path, self.run_name, self.extraction_name)

        return join(self.output_path, self.run_name, self.extraction_name, self.extra_model_folder)

    def get_file_content(self, file_name: str, default: Any = None) -> Any:
        path = Path(self.get_path(), file_name)
        if not path.exists():
            return default

        return json.loads(path.read_text())

    def save_content(self, file_name: str, content: Any, use_json: bool = True):
        path = Path(self.get_path(), file_name)

        if not exists(path.parent):
            os.makedirs(path.parent, exist_ok=True)

        if not use_json:
            path.write_text(str(content))
        elif type(content) == list:
            path.write_text(json.dumps([x.model_dump() for x in content]))
        else:
            path.write_text(json.dumps(content))

    def get_options_path(self):
        return Path(self.get_path(), OPTIONS_FILE_NAME)

    def is_old(self):
        path = self.get_path()
        return exists(path) and os.path.isdir(path) and os.path.getmtime(path) < (time() - (2 * 24 * 3600))

    @staticmethod
    def get_default():
        return ExtractionIdentifier(extraction_name="default")

    def clean_extractor_folder(self, method_name: str):
        if not os.path.exists(self.get_path()):
            return

        for name in os.listdir(self.get_path()):
            if name.strip().lower() == method_name.strip().lower():
                continue

            path = Path(self.get_path(), name)

            if path.is_file():
                continue

            for key_word_to_delete in ["setfit", "t5", "bert"]:
                if key_word_to_delete in name.lower():
                    shutil.rmtree(path, ignore_errors=True)
                    break

    def __str__(self):
        return f"{self.run_name} / {self.extraction_name}"
