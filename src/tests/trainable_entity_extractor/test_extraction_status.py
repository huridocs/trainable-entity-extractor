from pathlib import Path
from unittest import TestCase

from trainable_entity_extractor.data.ExtractionIdentifier import (
    ExtractionIdentifier,
    PROCESSING_FINISHED_FILE_NAME,
    METHOD_USED_FILE_NAME,
)
from trainable_entity_extractor.data.ExtractionStatus import ExtractionStatus


class TestExtractionStatus(TestCase):

    def test_no_model(self):
        extraction_identifier = ExtractionIdentifier(extraction_name="extraction_status_extractor")
        Path(extraction_identifier.get_path(), PROCESSING_FINISHED_FILE_NAME).unlink(missing_ok=True)
        Path(extraction_identifier.get_path(), METHOD_USED_FILE_NAME).unlink(missing_ok=True)
        extraction_status: ExtractionStatus = extraction_identifier.get_status()
        self.assertEqual(ExtractionStatus.NO_MODEL, extraction_status)

    def test_processing(self):
        extraction_identifier = ExtractionIdentifier(extraction_name="extraction_status_extractor")
        extraction_identifier.save_method_used("method_used")
        Path(extraction_identifier.get_path(), PROCESSING_FINISHED_FILE_NAME).unlink(missing_ok=True)
        extraction_status: ExtractionStatus = extraction_identifier.get_status()
        self.assertEqual(ExtractionStatus.PROCESSING, extraction_status)

    def test_ready(self):
        extraction_identifier = ExtractionIdentifier(extraction_name="extraction_status_extractor")
        extraction_identifier.save_method_used("method_used")
        extraction_identifier.save_content(PROCESSING_FINISHED_FILE_NAME, True)
        extraction_status: ExtractionStatus = extraction_identifier.get_status()
        self.assertEqual(ExtractionStatus.READY, extraction_status)

    def test_set_to_processing(self):
        extraction_identifier = ExtractionIdentifier(extraction_name="extraction_status_extractor")
        extraction_identifier.save_method_used("method_used")
        extraction_identifier.save_content(PROCESSING_FINISHED_FILE_NAME, True)
        extraction_identifier.set_extractor_to_processing()
        extraction_status: ExtractionStatus = extraction_identifier.get_status()
        self.assertEqual(ExtractionStatus.PROCESSING, extraction_status)
