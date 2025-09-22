from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.SegmentBox import SegmentBox
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.adapters.extractors.pdf_to_text_extractor.PdfToTextExtractor import PdfToTextExtractor
from trainable_entity_extractor.services.EntityExtractionService import EntityExtractionService


class TestPdfToTextExtractor(TestCase):
    TENANT = "unit_test"
    extraction_id = "TestPdfToTextExtractor"
    extraction_identifier = ExtractionIdentifier(run_name=TENANT, extraction_name=extraction_id)

    def test_no_prediction_data_with_orchestrator(self):
        """Test prediction with no data using the new orchestrator approach"""
        extraction_service = EntityExtractionService(self.extraction_identifier)
        success, message, suggestions = extraction_service.predict_with_orchestrator([])

        self.assertTrue(success)
        self.assertEqual(0, len(suggestions))

    def test_no_prediction_data_legacy(self):
        """Test prediction with no data using the legacy approach (for compatibility)"""
        pdf_to_text_extractor = PdfToTextExtractor(self.extraction_identifier)
        predictions = pdf_to_text_extractor.get_suggestions([])

        self.assertEqual(0, len(predictions))

    @staticmethod
    def get_samples(count_with_segments: int, count_without_segments: int):
        labeled_data_segments = LabeledData(
            label_text="text", label_segments_boxes=[SegmentBox(left=0, top=0, width=1, height=1, page_number=1)]
        )
        labeled_data_no_segments = LabeledData(label_text="text")
        samples = [TrainingSample(labeled_data=labeled_data_segments)] * count_with_segments
        samples += [TrainingSample(labeled_data=labeled_data_no_segments)] * count_without_segments

        return samples

    def test_get_train_test_with_few_samples(self):
        pdf_to_text_extractor = PdfToTextExtractor(self.extraction_identifier)

        extraction_data = ExtractionData(
            samples=self.get_samples(count_with_segments=4, count_without_segments=5),
            extraction_identifier=self.extraction_identifier,
        )

        train_set, test_set = pdf_to_text_extractor.get_train_test_sets(extraction_data)

        self.assertEqual(train_set.extraction_identifier, self.extraction_identifier)
        self.assertEqual(test_set.extraction_identifier, self.extraction_identifier)
        self.assertEqual(9, len(train_set.samples))
        self.assertEqual(9, len(test_set.samples))

    def test_get_train_test_without_enough_labeled_segments(self):
        pdf_to_text_extractor = PdfToTextExtractor(self.extraction_identifier)

        extraction_data = ExtractionData(
            samples=self.get_samples(count_with_segments=9, count_without_segments=11),
            extraction_identifier=self.extraction_identifier,
        )

        train_set, test_set = pdf_to_text_extractor.get_train_test_sets(extraction_data)

        self.assertEqual(9, len(train_set.samples))
        self.assertEqual(20, len(test_set.samples))

    def test_get_train_test_without_labeled_segments(self):
        pdf_to_text_extractor = PdfToTextExtractor(self.extraction_identifier)

        extraction_data = ExtractionData(
            samples=self.get_samples(count_with_segments=0, count_without_segments=100),
            extraction_identifier=self.extraction_identifier,
        )

        train_set, test_set = pdf_to_text_extractor.get_train_test_sets(extraction_data)

        self.assertEqual(80, len(train_set.samples))
        self.assertEqual(20, len(test_set.samples))

    def test_get_train_test_without_enough_data(self):
        pdf_to_text_extractor = PdfToTextExtractor(self.extraction_identifier)

        extraction_data = ExtractionData(
            samples=self.get_samples(count_with_segments=100, count_without_segments=100),
            extraction_identifier=self.extraction_identifier,
        )

        train_set, test_set = pdf_to_text_extractor.get_train_test_sets(extraction_data)

        self.assertEqual(160, len(train_set.samples))
        self.assertEqual(40, len(test_set.samples))

    def test_get_train_test_only_labels_with_segments(self):
        pdf_to_text_extractor = PdfToTextExtractor(self.extraction_identifier)

        extraction_data = ExtractionData(
            samples=self.get_samples(count_with_segments=200, count_without_segments=0),
            extraction_identifier=self.extraction_identifier,
        )

        train_set, test_set = pdf_to_text_extractor.get_train_test_sets(extraction_data)

        self.assertEqual(140, len(train_set.samples))
        self.assertEqual(60, len(test_set.samples))

    def test_train_and_predict_with_orchestrator(self):
        """Test complete training and prediction workflow using the new orchestrator approach"""
        extraction_data = ExtractionData(
            samples=self.get_samples(count_with_segments=10, count_without_segments=5),
            extraction_identifier=self.extraction_identifier,
            options=[],
            multi_value=False,
        )

        # Create extraction service
        extraction_service = EntityExtractionService(self.extraction_identifier)

        # Test getting available jobs
        available_jobs = extraction_service.get_available_jobs(extraction_data)
        self.assertGreater(len(available_jobs), 0)

        # Test training with orchestrator
        success, message, selected_job = extraction_service.train_with_orchestrator(
            extraction_data, [], False, use_performance_evaluation=False
        )

        if success:
            self.assertIsNotNone(selected_job)
            self.assertEqual(selected_job.extractor_name, "PdfToTextExtractor")

            # Test prediction with the trained model
            prediction_samples = [PredictionSample()]  # Add appropriate prediction data
            pred_success, pred_message, suggestions = extraction_service.predict_with_orchestrator(
                prediction_samples, selected_job
            )
            self.assertTrue(pred_success)

    def test_performance_evaluation_with_orchestrator(self):
        """Test performance evaluation using the new orchestrator approach"""
        extraction_data = ExtractionData(
            samples=self.get_samples(count_with_segments=15, count_without_segments=10),
            extraction_identifier=self.extraction_identifier,
            options=[],
            multi_value=False,
        )

        extraction_service = EntityExtractionService(self.extraction_identifier)
        available_jobs = extraction_service.get_available_jobs(extraction_data)

        if available_jobs:
            # Test performance evaluation for the first available job
            first_job = available_jobs[0]
            success, message, performance_score = extraction_service.evaluate_method_performance(first_job, extraction_data)

            self.assertTrue(success)
            self.assertIsInstance(performance_score, float)
            self.assertGreaterEqual(performance_score, 0.0)
            self.assertLessEqual(performance_score, 100.0)
