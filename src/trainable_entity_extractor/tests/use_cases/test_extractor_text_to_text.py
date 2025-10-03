import shutil
from unittest import TestCase

from trainable_entity_extractor.adapters.ExtractorLogger import ExtractorLogger
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.TextToTextExtractor import TextToTextExtractor
from trainable_entity_extractor.adapters.LocalExtractionDataRetriever import LocalExtractionDataRetriever
from trainable_entity_extractor.adapters.LocalModelStorage import LocalModelStorage
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.TrainUseCase import TrainUseCase
from trainable_entity_extractor.use_cases.PredictUseCase import PredictUseCase

extraction_id = "test_extractor_text_to_text"
extraction_identifier = ExtractionIdentifier(extraction_name=extraction_id)


class TestExtractorTextToText(TestCase):
    def setUp(self):
        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)
        self.data_retriever = LocalExtractionDataRetriever()
        self.model_storage = LocalModelStorage()
        self.extractors = [TextToTextExtractor]
        logger = ExtractorLogger()
        self.train_use_case = TrainUseCase(extractors=self.extractors, logger=logger)
        self.predict_use_case = PredictUseCase(extractors=self.extractors, logger=logger)

    def tearDown(self):
        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)

    def _create_and_train_model(self, method_name: str, extraction_data: ExtractionData) -> TrainableEntityExtractorJob:
        # Save extraction data
        self.data_retriever.save_extraction_data(extraction_identifier, extraction_data)

        # Get available jobs for training
        jobs = self.train_use_case.get_jobs(extraction_data)
        self.assertGreater(len(jobs), 0, "No training jobs available")

        extractor_job = [job for job in jobs if job.method_name == method_name][0]

        # Train the model
        success, message = self.train_use_case.train_one_method(extractor_job, extraction_data)
        self.assertTrue(success, f"Training failed: {message}")

        # Save the trained job
        self.model_storage.upload_model(extraction_identifier=extraction_identifier, extractor_job=extractor_job)

        return extractor_job

    def test_predictions_same_input_output(self):
        labeled_data = LabeledData(label_text="one", language_iso="en")
        sample = [TrainingSample(labeled_data=labeled_data, segment_selector_texts=["two"])]
        extraction_data = ExtractionData(samples=sample, extraction_identifier=extraction_identifier)

        method_name = "SameInputOutputMethod"
        extractor_job = self._create_and_train_model(method_name, extraction_data)

        texts = ["test 0", "test 1", "test 2"]
        predictions_samples = [PredictionSample.from_text(text, str(i)) for i, text in enumerate(texts)]

        self.data_retriever.save_prediction_data(extraction_identifier, predictions_samples)

        suggestions = self.predict_use_case.predict(extractor_job, predictions_samples)

        self.assertEqual(3, len(suggestions))
        self.assertEqual("default", suggestions[0].tenant)
        self.assertEqual(extraction_id, suggestions[0].id)
        self.assertEqual("0", suggestions[0].entity_name)
        self.assertEqual("test 0", suggestions[0].text)
        self.assertEqual("1", suggestions[1].entity_name)
        self.assertEqual("test 1", suggestions[1].text)
        self.assertEqual("2", suggestions[2].entity_name)
        self.assertEqual("test 2", suggestions[2].text)

    def test_predictions_from_source_text_in_labeled_data(self):
        samples = [TrainingSample(labeled_data=LabeledData(label_text="1", language_iso="en", source_text="foo 1"))]
        samples += [TrainingSample(labeled_data=LabeledData(label_text="2", language_iso="en", source_text="2 var"))]
        extraction_data = ExtractionData(samples=samples, extraction_identifier=extraction_identifier)

        # Train the model
        method_name = "RegexMethod"
        extractor_job = self._create_and_train_model(method_name, extraction_data)

        # Create prediction samples
        texts = ["test 0"]
        predictions_samples = [PredictionSample.from_text(text, str(i)) for i, text in enumerate(texts)]

        # Save prediction data
        self.data_retriever.save_prediction_data(extraction_identifier, predictions_samples)

        # Make predictions
        suggestions = self.predict_use_case.predict(extractor_job, predictions_samples)

        self.assertEqual(1, len(suggestions))
        self.assertEqual("0", suggestions[0].text)
