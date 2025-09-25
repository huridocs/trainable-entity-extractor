from unittest import TestCase
import os

from trainable_entity_extractor.adapters.ExtractorLogger import ExtractorLogger
from trainable_entity_extractor.config import APP_PATH
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.XmlFile import XmlFileUseCase
from trainable_entity_extractor.domain.SegmentationData import SegmentationData
from trainable_entity_extractor.domain.SegmentBox import SegmentBox
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.adapters.extractors.pdf_to_text_extractor.PdfToTextExtractor import PdfToTextExtractor
import shutil


class TestPdfToTextExtractor(TestCase):
    TENANT = "unit_test"
    extraction_id = "TestPdfToTextExtractor"
    extraction_identifier = ExtractionIdentifier(run_name=TENANT, extraction_name=extraction_id)

    def setUp(self):
        self.logger = ExtractorLogger()
        self.xml_file_path = str(APP_PATH / "trainable_entity_extractor" / "tests" / "test_files" / "test.xml")

    def tearDown(self):
        extraction_path = self.extraction_identifier.get_path()
        if os.path.exists(extraction_path):
            shutil.rmtree(extraction_path, ignore_errors=True)

    def test_pdf_to_text_extractor_with_real_data(self):
        """Test PdfToTextExtractor training with real XML data to extract '170221'"""

        # Create multiple training samples using the same XML file
        training_samples = []

        for i in range(5):  # Create 5 training samples for better training
            # Create labeled data for extracting "170221"
            labeled_data = LabeledData(
                tenant=self.TENANT,
                id=f"sample_{i}",
                xml_file_name="test.xml",
                entity_name="document_number",
                language_iso="en",
                label_text="170221",
                empty_value=False,
                source_text="",
                page_width=612.0,
                page_height=792.0,
                xml_segments_boxes=[],
                label_segments_boxes=[
                    # Create a bounding box around where "170221" appears in the XML
                    # Based on the XML, "170221" appears in text with top="739" left="65"
                    SegmentBox(left=65, top=739, width=103, height=11, page_number=1)
                ],
            )

            # Create XmlFileUseCase for the test XML file
            xml_file_use_case = XmlFileUseCase(
                extraction_identifier=self.extraction_identifier, to_train=True, xml_file_name="test.xml"
            )
            xml_file_use_case.xml_file_path = self.xml_file_path

            # Create segmentation data
            segmentation_data = SegmentationData.from_labeled_data(labeled_data)

            # Create PdfData from XML file
            pdf_data = PdfData.from_xml_file(xml_file_use_case, segmentation_data)

            # Create training sample
            training_sample = TrainingSample(labeled_data=labeled_data, pdf_data=pdf_data)
            training_samples.append(training_sample)

        # Create extraction data
        extraction_data = ExtractionData(extraction_identifier=self.extraction_identifier, samples=training_samples)

        # Create PdfToTextExtractor
        extractor = PdfToTextExtractor(extraction_identifier=self.extraction_identifier, logger=self.logger)

        # Check if the extractor can be used with this data
        self.assertTrue(extractor.can_be_used(extraction_data))

        # Create training job for PdfToTextSegmentSelector_RegexMethod
        method_name = "PdfToTextSegmentSelectorRegexMethod"
        job = TrainableEntityExtractorJob(
            run_name=self.TENANT,
            extraction_name=self.extraction_id,
            extractor_name="PdfToTextExtractor",
            method_name=method_name,
            gpu_needed=False,
            timeout=300,  # 5 minutes timeout for training
        )

        # Train the model
        success, error_msg = extractor.train_one_method(job, extraction_data)
        self.assertTrue(success, f"Training failed: {error_msg}")

        # Test prediction with get_suggestions
        # Create a prediction sample using the same XML file
        prediction_xml_file_use_case = XmlFileUseCase(
            extraction_identifier=self.extraction_identifier,
            to_train=False,  # This is for prediction, not training
            xml_file_name="test.xml",
        )
        prediction_xml_file_use_case.xml_file_path = self.xml_file_path

        # Create prediction data (without label information)
        prediction_labeled_data = LabeledData(
            tenant=self.TENANT,
            id="prediction_sample",
            xml_file_name="test.xml",
            entity_name="document_number",
            language_iso="en",
            label_text="",  # Empty for prediction
            empty_value=False,
            source_text="",
            page_width=612.0,
            page_height=792.0,
            xml_segments_boxes=[],
            label_segments_boxes=[],  # Empty for prediction
        )

        prediction_segmentation_data = SegmentationData.from_labeled_data(prediction_labeled_data)
        prediction_pdf_data = PdfData.from_xml_file(prediction_xml_file_use_case, prediction_segmentation_data)

        # Create prediction samples data
        from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
        from trainable_entity_extractor.domain.PredictionSample import PredictionSample

        prediction_sample = PredictionSample(entity_name="document_number", pdf_data=prediction_pdf_data)

        prediction_samples_data = PredictionSamplesData(prediction_samples=[prediction_sample])

        # Get suggestions from the trained model
        suggestions = extractor.get_suggestions(method_name, prediction_samples_data)

        # Verify that we got suggestions
        self.assertTrue(len(suggestions) > 0, "Should have at least one suggestion")

        # Check that the suggestion contains the expected text "170221"
        suggestion = suggestions[0]
        self.assertEqual(suggestion.entity_name, "document_number")
        self.assertEqual(suggestion.text, "170221", f"Expected '170221' but got '{suggestion.text}'")
        self.assertEqual(
            suggestion.segment_text,
            (
                '<p class="ix_matching_paragraph">21-02065 (E) <span '
                'class="ix_match">170221</span></p><p '
                'class="ix_adjacent_paragraph">*2102065*</p>'
            ),
        )
