import unittest
from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.Gemini.GeminiTextMethod import (
    GeminiTextMethod,
)
from trainable_entity_extractor.domain.ExtractionData import ExtractionData


class TestGeminiTextMethodWithRealAPI(TestCase):
    @unittest.skip("Requires real Gemini API key")
    def test_gemini(self):
        extraction_identifier = ExtractionIdentifier(run_name="unit_test", extraction_name="gemini_text_test")
        gemini_text_method = GeminiTextMethod(extraction_identifier)

        extraction_data = ExtractionData(
            samples=[
                TrainingSample(
                    labeled_data=LabeledData(label_text="This is a test output", source_text="This is a test input"),
                ),
                TrainingSample(
                    labeled_data=LabeledData(label_text="Same for output", source_text="Same for input"),
                ),
            ],
            extraction_identifier=extraction_identifier,
        )

        gemini_text_method.train(extraction_data)

        prediction_samples_data = PredictionSamplesData(
            prediction_samples=[PredictionSample(source_text="This is other input")], options=[], multi_value=False
        )
        predictions = gemini_text_method.predict(prediction_samples_data)

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0], "This is other output")

    @unittest.skip("Requires real Gemini API key")
    def test_gemini_with_other_data(self):
        extraction_identifier = ExtractionIdentifier(run_name="unit_test", extraction_name="gemini_text_other_data")
        gemini_text_method = GeminiTextMethod(extraction_identifier)

        extraction_data = ExtractionData(
            samples=[
                TrainingSample(
                    labeled_data=LabeledData(label_text="Output A", language_iso="en"),
                    segment_selector_texts=["Input A"],
                ),
                TrainingSample(
                    labeled_data=LabeledData(label_text="Output B", language_iso="en"),
                    segment_selector_texts=["Input B"],
                ),
            ],
            extraction_identifier=extraction_identifier,
        )

        gemini_text_method.train(extraction_data)

        prediction_samples_data = PredictionSamplesData(
            prediction_samples=[PredictionSample(source_text="Input B")], options=[], multi_value=False
        )
        predictions = gemini_text_method.predict(prediction_samples_data)

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0], "Output B")
