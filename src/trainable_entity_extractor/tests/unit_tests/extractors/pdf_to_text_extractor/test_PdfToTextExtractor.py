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

    def _create_training_samples(self, num_samples=5):
        training_samples = []

        for i in range(num_samples):
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
                label_segments_boxes=[SegmentBox(left=65, top=739, width=103, height=11, page_number=1)],
            )

            xml_file_use_case = XmlFileUseCase(
                extraction_identifier=self.extraction_identifier, to_train=True, xml_file_name="test.xml"
            )
            xml_file_use_case.xml_file_path = self.xml_file_path

            segmentation_data = SegmentationData.from_labeled_data(labeled_data)

            pdf_data = PdfData.from_xml_file(xml_file_use_case, segmentation_data)

            training_sample = TrainingSample(labeled_data=labeled_data, pdf_data=pdf_data)
            training_samples.append(training_sample)

        return training_samples

    def _create_prediction_sample(self):
        prediction_xml_file_use_case = XmlFileUseCase(
            extraction_identifier=self.extraction_identifier,
            to_train=False,
            xml_file_name="test.xml",
        )
        prediction_xml_file_use_case.xml_file_path = self.xml_file_path

        prediction_labeled_data = LabeledData(
            tenant=self.TENANT,
            id="prediction_sample",
            xml_file_name="test.xml",
            entity_name="document_number",
            language_iso="en",
            label_text="",
            empty_value=False,
            source_text="",
            page_width=612.0,
            page_height=792.0,
            xml_segments_boxes=[],
            label_segments_boxes=[],
        )

        prediction_segmentation_data = SegmentationData.from_labeled_data(prediction_labeled_data)
        prediction_pdf_data = PdfData.from_xml_file(prediction_xml_file_use_case, prediction_segmentation_data)

        from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
        from trainable_entity_extractor.domain.PredictionSample import PredictionSample

        prediction_sample = PredictionSample(entity_name="document_number", pdf_data=prediction_pdf_data)
        prediction_samples_data = PredictionSamplesData(prediction_samples=[prediction_sample])

        return prediction_samples_data

    def _train_and_test_method(self, method_name):
        training_samples = self._create_training_samples()

        extraction_data = ExtractionData(extraction_identifier=self.extraction_identifier, samples=training_samples)

        extractor = PdfToTextExtractor(extraction_identifier=self.extraction_identifier, logger=self.logger)

        self.assertTrue(extractor.can_be_used(extraction_data))

        job = TrainableEntityExtractorJob(
            run_name=self.TENANT,
            extraction_name=self.extraction_id,
            extractor_name="PdfToTextExtractor",
            method_name=method_name,
            gpu_needed=False,
            timeout=300,
        )

        success, error_msg = extractor.train_one_method(job, extraction_data)
        self.assertTrue(success, f"Training failed: {error_msg}")

        prediction_samples_data = self._create_prediction_sample()

        suggestions = extractor.get_suggestions(method_name, prediction_samples_data)

        self.assertTrue(len(suggestions) > 0, "Should have at least one suggestion")

        suggestion = suggestions[0]
        self.assertEqual(suggestion.entity_name, "document_number")
        self.assertEqual("170221", suggestion.text, f"Expected '170221' but got '{suggestion.text}'")
        self.assertEqual(
            suggestion.segment_text,
            (
                '<p class="ix_matching_paragraph">21-02065 (E) <span '
                'class="ix_match">170221</span></p><p '
                'class="ix_adjacent_paragraph">*2102065*</p>'
            ),
        )

        return method_name, len(training_samples)

    def test_pdf_to_text_extractor_with_real_data(self):
        method_name, num_samples = self._train_and_test_method("PdfToTextSegmentSelectorRegexMethod")
        print(f"✅ Successfully trained and tested {method_name} with {num_samples} samples")

    def test_pdf_to_text_extractor_with_fast_segment_selector(self):
        method_name, num_samples = self._train_and_test_method("PdfToTextFastSegmentSelectorRegexSubtractionMethod")
        print(f"✅ Successfully trained and tested {method_name} with {num_samples} samples")
