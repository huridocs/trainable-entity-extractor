import os
import pathlib
from os.path import join

from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier


class XmlFile:
    def __init__(self, extraction_identifier: ExtractionIdentifier, to_train: bool, xml_file_name: str):
        self.extraction_identifier = extraction_identifier
        self.to_train = to_train
        self.xml_file_name = xml_file_name
        self.xml_file = None
        self.xml_folder_path = XmlFile.get_xml_folder_path(extraction_identifier, to_train)
        self.xml_file_path = os.path.join(self.xml_folder_path, self.xml_file_name)

    def save(self, file: bytes):
        if not os.path.exists(self.xml_folder_path):
            os.makedirs(self.xml_folder_path)

        file_path = pathlib.Path(f"{self.xml_folder_path}/{self.xml_file_name}")
        file_path.write_bytes(file)

    @staticmethod
    def get_xml_folder_path(extraction_identifier: ExtractionIdentifier, to_train: bool) -> str:
        if to_train:
            return join(extraction_identifier.get_path(), "xml_to_train")

        return join(extraction_identifier.get_path(), "xml_to_predict")
