import os
from os.path import join
from pathlib import Path

from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier


class XmlFile:
    def __init__(self, extraction_identifier: ExtractionIdentifier, to_train: bool, xml_file_name: str):
        self.extraction_identifier = extraction_identifier
        self.to_train = to_train
        self.xml_file_name = xml_file_name
        self.xml_file = None
        self.xml_folder_path = self.get_xml_folder_path(to_train)
        self.xml_file_path = join(self.xml_folder_path, self.xml_file_name)

    def save(self, file: bytes):
        os.makedirs(self.xml_folder_path, exist_ok=True)
        file_path = Path(f"{self.xml_folder_path}/{self.xml_file_name}")
        file_path.write_bytes(file)

    def get_xml_folder_path(self, to_train: bool) -> str:
        if to_train:
            return join(self.extraction_identifier.get_path(), "xml_to_train")

        return join(self.extraction_identifier.get_path(), "xml_to_predict")
