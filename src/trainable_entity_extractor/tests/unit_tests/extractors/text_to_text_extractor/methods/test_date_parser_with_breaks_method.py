from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.DateParserWithBreaksMethod import (
    DateParserWithBreaksMethod,
)


class TestDateParserWithBreaksMethod(TestCase):
    def setUp(self):
        self.extraction_identifier = ExtractionIdentifier(run_name="unit_test", extraction_name="test")

    def test_predict(self):
        sample = TrainingSample(
            labeled_data=LabeledData(label_text="1982-06-05", language_iso="en", source_text="5 Jun 1982")
        )

        extraction_data = ExtractionData(
            samples=[sample for _ in range(6)], extraction_identifier=self.extraction_identifier
        )
        date_parser_method = DateParserWithBreaksMethod(self.extraction_identifier)

        date_parser_method.train(extraction_data)

        text_1 = "ORDER OF THE INTER-AMERICAN COURT OF HUMAN RIGHTS 1 OF FEBRUARY 9, 2006"
        text_2 = "ORDER OF THE INTER-AMERICAN COURT OF HUMAN RIGHTS 1 OF MARCH 10, 2007"

        prediction_data = PredictionSamplesData(
            prediction_samples=[PredictionSample(source_text=text_1), PredictionSample(source_text=text_2)],
            options=[],
            multi_value=False,
        )
        predictions = date_parser_method.predict(prediction_data)

        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 2)

    def test_method_initialization(self):
        """Test that DateParserWithBreaksMethod can be properly initialized with real instances"""
        date_parser_method = DateParserWithBreaksMethod(self.extraction_identifier)

        self.assertIsNotNone(date_parser_method)
        self.assertEqual(date_parser_method.extraction_identifier, self.extraction_identifier)
