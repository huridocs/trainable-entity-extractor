from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.DateParserMethod import DateParserMethod
from trainable_entity_extractor.adapters.ExtractorLogger import ExtractorLogger


class TestDateParserMethod(TestCase):
    def setUp(self):
        self.extraction_identifier = ExtractionIdentifier(run_name="unit_test", extraction_name="date_test")

    def test_performance(self):
        sample = TrainingSample(
            labeled_data=LabeledData(label_text="1981-05-13", language_iso="en", source_text="13 May 1981")
        )

        extraction_data = ExtractionData(
            samples=[sample for _ in range(6)], extraction_identifier=self.extraction_identifier
        )
        date_parser_method = DateParserMethod(self.extraction_identifier)
        performance = date_parser_method.get_performance(extraction_data, extraction_data)
        self.assertIsInstance(performance, (int, float))
        self.assertGreaterEqual(performance, 0)

    def test_predict(self):
        sample = TrainingSample(
            labeled_data=LabeledData(label_text="1982-06-05", language_iso="en", source_text="5 Jun 1982")
        )

        extraction_data = ExtractionData(
            samples=[sample for _ in range(6)], extraction_identifier=self.extraction_identifier
        )
        date_parser_method = DateParserMethod(self.extraction_identifier)

        date_parser_method.train(extraction_data)

        prediction_data = PredictionSamplesData(
            prediction_samples=[PredictionSample(source_text="5 Jun 1982")], options=[], multi_value=False
        )
        predictions = date_parser_method.predict(prediction_data)
        self.assertEqual(["1982-06-05"], predictions)

    def test_predict_multiple_dates_spanish(self):
        text = "Informe no. 52/16 caso 12.521 fondo Maria Laura órdenes guerra y otros Chile 30 de noviembre de 2016"
        sample = TrainingSample(labeled_data=LabeledData(label_text="2016-11-30", language_iso="es", source_text=text))

        extraction_data = ExtractionData(
            samples=[sample for _ in range(6)], extraction_identifier=self.extraction_identifier
        )
        date_parser_method = DateParserMethod(self.extraction_identifier)

        date_parser_method.train(extraction_data)

        prediction_data = PredictionSamplesData(
            prediction_samples=[PredictionSample(source_text=text)], options=[], multi_value=False
        )
        predictions = date_parser_method.predict(prediction_data)
        self.assertEqual(["2016-11-30"], predictions)

    def test_method_initialization(self):
        """Test that DateParserMethod can be properly initialized with real instances"""
        date_parser_method = DateParserMethod(self.extraction_identifier)

        self.assertIsNotNone(date_parser_method)
        self.assertEqual(date_parser_method.extraction_identifier, self.extraction_identifier)
