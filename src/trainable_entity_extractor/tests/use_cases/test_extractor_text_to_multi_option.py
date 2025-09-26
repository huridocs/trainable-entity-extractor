import shutil
from unittest import TestCase

from trainable_entity_extractor.adapters.ExtractorLogger import ExtractorLogger
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.TextToMultiOptionExtractor import (
    TextToMultiOptionExtractor,
)
from trainable_entity_extractor.adapters.LocalExtractionDataRetriever import LocalExtractionDataRetriever
from trainable_entity_extractor.adapters.LocalModelStorage import LocalModelStorage
from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.TrainUseCase import TrainUseCase
from trainable_entity_extractor.use_cases.PredictUseCase import PredictUseCase

extraction_id = "test_extractor_text_to_multi_option"
extraction_identifier = ExtractionIdentifier(extraction_name=extraction_id)


class TestExtractorTextToMultiOption(TestCase):
    def setUp(self):
        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)
        self.data_retriever = LocalExtractionDataRetriever()
        self.model_storage = LocalModelStorage()
        self.extractors = [TextToMultiOptionExtractor]
        logger = ExtractorLogger()
        self.train_use_case = TrainUseCase(extractors=self.extractors, logger=logger)
        self.predict_use_case = PredictUseCase(extractors=self.extractors, logger=logger)

    def tearDown(self):
        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)

    def _create_and_train_model(self, extraction_data: ExtractionData):
        # Save extraction data
        self.data_retriever.save_extraction_data(extraction_identifier, extraction_data)

        # Get available jobs for training
        jobs = self.train_use_case.get_jobs(extraction_data)
        self.assertGreater(len(jobs), 0, "No training jobs available")

        # Use the first available job (typically a fuzzy matching method for multi-option)
        extractor_job = [job for job in jobs if job.method_name == "TextFuzzyAll100"][0]

        # Train the model
        success, message = self.train_use_case.train_one_method(extractor_job, extraction_data)
        self.assertTrue(success, f"Training failed: {message}")

        # Save the trained job
        self.model_storage.upload_model(extraction_identifier, extractor_job)

        return extractor_job

    def test_get_text_multi_option_suggestions(self):
        options = [Option(id="1", label="abc"), Option(id="2", label="dfg"), Option(id="3", label="hij")]

        values_1 = [Option(id="1", label="abc"), Option(id="2", label="dfg")]
        labeled_data_1 = LabeledData(language_iso="en", values=values_1, source_text="foo abc dfg")

        values_2 = [Option(id="2", label="dfg"), Option(id="3", label="hij")]
        labeled_data_2 = LabeledData(language_iso="en", values=values_2, source_text="foo dfg hij")

        sample = [TrainingSample(labeled_data=labeled_data_1), TrainingSample(labeled_data=labeled_data_2)]
        extraction_data = ExtractionData(
            samples=sample, extraction_identifier=extraction_identifier, multi_value=True, options=options
        )

        # Train the model
        extractor_job = self._create_and_train_model(extraction_data)

        # Create prediction samples using PredictionSamples
        prediction_samples_list = [PredictionSample.from_text("foo var dfg hij foo var", "0")]

        # Save prediction data
        self.data_retriever.save_prediction_data(extraction_identifier, prediction_samples_list)

        # Make predictions
        suggestions = self.predict_use_case.predict(extractor_job, prediction_samples_list)

        self.assertEqual(1, len(suggestions))
        self.assertEqual(extraction_id, suggestions[0].id)
        self.assertEqual(
            [
                Value(id="2", label="dfg", segment_text="foo var dfg hij foo var"),
                Value(id="3", label="hij", segment_text="foo var dfg hij foo var"),
            ],
            suggestions[0].values,
        )
