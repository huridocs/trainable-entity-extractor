from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.RegexMethod import RegexMethod
from trainable_entity_extractor.adapters.ExtractorLogger import ExtractorLogger


class TestRegexMethod(TestCase):
    def setUp(self):
        self.extraction_identifier = ExtractionIdentifier(run_name="unit_test", extraction_name="test")

    def test_performance_100(self):
        sample = TrainingSample(labeled_data=LabeledData(label_text="12", language_iso="en", source_text="one 12"))

        extraction_data = ExtractionData(
            samples=[sample for _ in range(6)], extraction_identifier=self.extraction_identifier
        )
        regex_method = RegexMethod(self.extraction_identifier)

        performance = regex_method.get_performance(extraction_data, extraction_data)
        self.assertIsInstance(performance, (int, float))
        self.assertGreaterEqual(performance, 0)

    def test_performance_0(self):
        sample = TrainingSample(labeled_data=LabeledData(label_text="12", language_iso="en", source_text="one two"))

        extraction_data = ExtractionData(
            samples=[sample for _ in range(6)], extraction_identifier=self.extraction_identifier
        )
        regex_method = RegexMethod(self.extraction_identifier)

        performance = regex_method.get_performance(extraction_data, extraction_data)
        self.assertIsInstance(performance, (int, float))
        self.assertGreaterEqual(performance, 0)

    def test_performance_mixed(self):
        sample_1 = TrainingSample(labeled_data=LabeledData(label_text="12", language_iso="en", source_text="one 12"))
        sample_2 = TrainingSample(labeled_data=LabeledData(label_text="no regex", language_iso="en", source_text="one"))

        samples = [sample_1] * 3 + [sample_2] * 1
        extraction_data = ExtractionData(samples=samples, extraction_identifier=self.extraction_identifier)
        regex_method = RegexMethod(self.extraction_identifier)

        performance = regex_method.get_performance(extraction_data, extraction_data)
        self.assertIsInstance(performance, (int, float))
        self.assertGreaterEqual(performance, 0)
