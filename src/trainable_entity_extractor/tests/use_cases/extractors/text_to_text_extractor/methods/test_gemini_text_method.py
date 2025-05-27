from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.Gemini.GeminiTextMethod import (
    GeminiTextMethod,
)
from trainable_entity_extractor.domain.ExtractionData import ExtractionData


class TestGeminiTextMethodWithRealAPI(TestCase):
    def test_train(self):
        extraction_identifier = ExtractionIdentifier(extraction_name="gemini_text_test")
        gemini_text_method = GeminiTextMethod(extraction_identifier)

        extraction_data = ExtractionData(
            samples=[
                TrainingSample(
                    labeled_data=LabeledData(label_text="This is a test output", language_iso="en"),
                    segment_selector_texts=["This is a test input"],
                ),
                TrainingSample(
                    labeled_data=LabeledData(label_text="Same for output", language_iso="en"),
                    segment_selector_texts=["Same for input"],
                ),
            ],
            extraction_identifier=extraction_identifier,
        )

        # gemini_text_method.train(extraction_data)
        predictions = gemini_text_method.predict([PredictionSample(source_text="This is other input")])

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0], "This is other output")
