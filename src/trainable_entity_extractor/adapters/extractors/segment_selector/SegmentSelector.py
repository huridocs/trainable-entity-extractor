import shutil
import lightgbm as lgb

from os import makedirs
from os.path import join, exists
from pathlib import Path

from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.adapters.extractors.segment_selector.SegmentSelectorBase import SegmentSelectorBase
from trainable_entity_extractor.adapters.extractors.segment_selector.methods.lightgbm_frequent_words.LightgbmFrequentWords import (
    LightgbmFrequentWords,
)


class SegmentSelector(SegmentSelectorBase):
    def __init__(self, extraction_identifier: ExtractionIdentifier, method_name: str = ""):
        super().__init__(extraction_identifier, method_name)
        self.model_path = join(self.extraction_identifier.get_path(), "segment_predictor_model", "model.model")
        self.model = self.load_model()

    def load_model(self):
        if exists(self.model_path):
            return lgb.Booster(model_file=self.model_path)

        return None

    def prepare_model_folder(self):
        shutil.rmtree(Path(self.model_path).parent, ignore_errors=True)

        model_path = self.model_path

        if not exists(Path(model_path).parent):
            makedirs(Path(model_path).parent)

        return model_path

    def create_model(self, pdfs_data: list[PdfData]) -> (bool, str):
        if Path(self.model_path).exists():
            return True, ""

        valid_pdf_data = self.get_valid_pdfs_data(pdfs_data)

        if not valid_pdf_data:
            return False, "No data to create model, no segments"

        self.model = LightgbmFrequentWords().create_model(valid_pdf_data, self.model_path)

        if not self.model:
            return False, "No data to create model, no model created"

        self.model.save_model(self.model_path, num_iteration=self.model.best_iteration)
        return True, ""

    @staticmethod
    def get_valid_pdfs_data(pdfs_data: list[PdfData]) -> list[PdfData]:
        valid_pdf_data = list()
        for pdf_data in pdfs_data:
            if not pdf_data.pdf_features or not pdf_data.pdf_data_segments:
                continue

            valid_pdf_data.append(pdf_data)
        return valid_pdf_data

    def set_extraction_segments(self, pdfs_data: list[PdfData]):
        if not self.model:
            return

        predictions = LightgbmFrequentWords().predict(self.model, pdfs_data, self.model_path)
        index = 0
        for pdf_metadata in pdfs_data:
            for segment in pdf_metadata.pdf_data_segments:
                segment.ml_label = 1 if predictions[index] > 0.5 else 0
                index += 1

    def get_predictions_for_performance(self, training_set: list[PdfData], test_set: list[PdfData]) -> list[int]:
        self.create_model(training_set)
        self.set_extraction_segments(test_set)
        return [segment.ml_label for pdf_data in test_set for segment in pdf_data.pdf_data_segments]
