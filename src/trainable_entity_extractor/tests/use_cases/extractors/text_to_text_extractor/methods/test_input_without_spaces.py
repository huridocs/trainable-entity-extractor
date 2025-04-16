from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.InputWithoutSpaces import (
    InputWithoutSpaces,
)

extraction_identifier = ExtractionIdentifier(run_name="test", extraction_name="test")


class TestInputWithoutSpaces(TestCase):
    def test_performance_100(self):
        sample = TrainingSample(
            labeled_data=LabeledData(label_text="abc", language_iso="en"), segment_selector_texts=["a b c"]
        )

        extraction_data = ExtractionData(samples=[sample], extraction_identifier=extraction_identifier)

        same_input_output_method = InputWithoutSpaces(extraction_identifier)
        self.assertEqual(100, same_input_output_method.performance(extraction_data, extraction_data))

    def test_performance_50(self):
        sample_1 = TrainingSample(
            labeled_data=LabeledData(label_text="abc", language_iso="en"), segment_selector_texts=["a b ", "c"]
        )
        sample_2 = TrainingSample(
            labeled_data=LabeledData(label_text="2", language_iso="en"), segment_selector_texts=["a", " b c"]
        )

        extraction_data = ExtractionData(samples=[sample_1] + [sample_2], extraction_identifier=extraction_identifier)

        same_input_output_method = InputWithoutSpaces(extraction_identifier)

        self.assertEqual(50, same_input_output_method.performance(extraction_data, extraction_data))

    def test_predict(self):
        same_input_output_method = InputWithoutSpaces(extraction_identifier)
        predictions = same_input_output_method.predict([PredictionSample.from_text(" test 1 4 foo ")])

        self.assertEqual(["test14foo"], predictions)
