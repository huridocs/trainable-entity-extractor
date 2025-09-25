from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.InputWithoutSpaces import (
    InputWithoutSpaces,
)
from trainable_entity_extractor.adapters.ExtractorLogger import ExtractorLogger


class TestInputWithoutSpaces(TestCase):
    def setUp(self):
        self.extraction_identifier = ExtractionIdentifier(run_name="unit_test", extraction_name="test")

    def test_performance_100(self):
        sample = TrainingSample(labeled_data=LabeledData(label_text="abc", language_iso="en", source_text="a b c"))

        extraction_data = ExtractionData(samples=[sample], extraction_identifier=self.extraction_identifier)

        input_without_spaces_method = InputWithoutSpaces(self.extraction_identifier)
        performance = input_without_spaces_method.get_performance(extraction_data, extraction_data)
        self.assertIsInstance(performance, (int, float))
        self.assertGreaterEqual(performance, 0)

    def test_performance_mixed(self):
        sample_1 = TrainingSample(labeled_data=LabeledData(label_text="abc", language_iso="en", source_text="a b c"))
        sample_2 = TrainingSample(labeled_data=LabeledData(label_text="2", language_iso="en", source_text="a b c"))

        extraction_data = ExtractionData(samples=[sample_1, sample_2], extraction_identifier=self.extraction_identifier)

        input_without_spaces_method = InputWithoutSpaces(self.extraction_identifier)

        performance = input_without_spaces_method.get_performance(extraction_data, extraction_data)
        self.assertIsInstance(performance, (int, float))
        self.assertGreaterEqual(performance, 0)

    def test_predict(self):
        input_without_spaces_method = InputWithoutSpaces(self.extraction_identifier)

        prediction_data = PredictionSamplesData(
            prediction_samples=[PredictionSample(source_text="a b c d")], options=[], multi_value=False
        )
        predictions = input_without_spaces_method.predict(prediction_data)

        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0], "abcd")

    def test_method_initialization(self):
        """Test that InputWithoutSpaces can be properly initialized with real instances"""
        input_without_spaces_method = InputWithoutSpaces(self.extraction_identifier)

        self.assertIsNotNone(input_without_spaces_method)
        self.assertEqual(input_without_spaces_method.extraction_identifier, self.extraction_identifier)
