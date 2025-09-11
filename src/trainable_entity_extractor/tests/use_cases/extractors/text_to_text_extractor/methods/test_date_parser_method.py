from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.DateParserMethod import DateParserMethod

extraction_identifier = ExtractionIdentifier(run_name="unit_test", extraction_name="date_test")


class TestDateParserMethod(TestCase):
    def test_performance(self):
        sample = TrainingSample(
            labeled_data=LabeledData(label_text="1981-05-13", language_iso="en"), segment_selector_texts=["13 May 1981"]
        )

        extraction_data = ExtractionData(samples=[sample for _ in range(6)], extraction_identifier=extraction_identifier)
        date_parser_method = DateParserMethod(extraction_identifier)
        self.assertEqual(100, date_parser_method.get_performance(extraction_data, extraction_data))

    def test_predict(self):
        sample = TrainingSample(
            labeled_data=LabeledData(label_text="1982-06-05", language_iso="en"), segment_selector_texts=["5 Jun 1982"]
        )

        extraction_data = ExtractionData(samples=[sample for _ in range(6)], extraction_identifier=extraction_identifier)
        date_parser_method = DateParserMethod(extraction_identifier)

        date_parser_method.train(extraction_data)

        predictions = date_parser_method.predict([PredictionSample.from_text("5 Jun 1982")])
        self.assertEqual(["1982-06-05"], predictions)

    def test_predict_multiple_dates_spanish(self):
        text = "Informe no. 52/16 caso 12.521 fondo Maria Laura órdenes guerra y otros Chile 30 de noviembre de 2016"
        sample = TrainingSample(
            labeled_data=LabeledData(label_text="2016-11-30", language_iso="es"), segment_selector_texts=[text]
        )

        extraction_data = ExtractionData(samples=[sample for _ in range(6)], extraction_identifier=extraction_identifier)
        date_parser_method = DateParserMethod(extraction_identifier)

        date_parser_method.train(extraction_data)

        predictions = date_parser_method.predict([PredictionSample.from_text(text)])
        self.assertEqual(["2016-11-30"], predictions)

    def test_performance_multiple_tags(self):
        sample = TrainingSample(
            labeled_data=LabeledData(label_text="1981-05-13", language_iso="es"), segment_selector_texts=["13 May", "1981"]
        )

        extraction_data = ExtractionData(samples=[sample for _ in range(6)], extraction_identifier=extraction_identifier)
        date_parser_method = DateParserMethod(extraction_identifier)

        self.assertEqual(100, date_parser_method.get_performance(extraction_data, extraction_data))
