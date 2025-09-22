from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.GlinerDateParserMethod import (
    GlinerDateParserMethod,
)

extraction_identifier = ExtractionIdentifier(run_name="unit_test", extraction_name="date_test")


class TestGlinerDateMethod(TestCase):
    def test_predict(self):
        gliner_method = GlinerDateParserMethod(extraction_identifier)

        predictions = gliner_method.predict([PredictionSample.from_text("5 Jun 1982")])
        self.assertEqual(["1982-06-05"], predictions)

    def test_predict_special_character(self):
        gliner_method = GlinerDateParserMethod(extraction_identifier)

        predictions = gliner_method.predict([PredictionSample.from_text("SENTENÇA DE 1° DE JULHO DE 2009")])
        self.assertEqual(["2009-07-01"], predictions)

    def test_predict_portuguese(self):
        gliner_method = GlinerDateParserMethod(extraction_identifier)

        predictions = gliner_method.predict([PredictionSample.from_text("SENTENÇA DE 1° DE MARÇO DE 2010")])
        self.assertEqual(["2010-03-01"], predictions)

    def test_predict_multiple_dates_spanish(self):
        text = "Informe no. 52/16 caso 12.521 fondo Maria Laura órdenes guerra y otros Chile 30 de noviembre de 2016"
        sample = TrainingSample(
            labeled_data=LabeledData(label_text="2016-11-30", language_iso="es"), segment_selector_texts=[text]
        )

        extraction_data = ExtractionData(samples=[sample for _ in range(6)], extraction_identifier=extraction_identifier)
        gliner_method = GlinerDateParserMethod(extraction_identifier)

        gliner_method.train(extraction_data)

        predictions = gliner_method.predict([PredictionSample.from_text(text)])
        self.assertEqual(["2016-11-30"], predictions)

    def test_performance_multiple_tags(self):
        sample = TrainingSample(
            labeled_data=LabeledData(label_text="1981-05-13", language_iso="es"), segment_selector_texts=["13 May", "1981"]
        )

        extraction_data = ExtractionData(samples=[sample for _ in range(6)], extraction_identifier=extraction_identifier)
        gliner_method = GlinerDateParserMethod(extraction_identifier)

        self.assertEqual(100, gliner_method.get_performance(extraction_data, extraction_data))

    def test_is_valid_execution_file_functionality(self):
        """Test that IS_VALID_EXECUTION_FILE_NAME properly controls prediction behavior"""
        # Test with valid date samples - should not create invalid execution flag
        valid_sample = TrainingSample(
            labeled_data=LabeledData(label_text="2023-01-15", language_iso="en"),
            segment_selector_texts=["January 15, 2023"],
        )

        valid_extraction_data = ExtractionData(
            samples=[valid_sample for _ in range(6)], extraction_identifier=extraction_identifier
        )
        gliner_method = GlinerDateParserMethod(extraction_identifier)

        # Train with valid data
        gliner_method.train(valid_extraction_data)

        # Should make normal predictions
        predictions = gliner_method.predict([PredictionSample.from_text("March 10, 2024")])
        self.assertNotEqual([""], predictions)  # Should not return empty string

        # Test with invalid date samples - should create invalid execution flag
        invalid_sample = TrainingSample(
            labeled_data=LabeledData(label_text="not a date", language_iso="en"),
            segment_selector_texts=["some random text"],
        )

        invalid_extraction_data = ExtractionData(
            samples=[invalid_sample for _ in range(6)], extraction_identifier=extraction_identifier
        )

        # Train with invalid data
        gliner_method.train(invalid_extraction_data)

        # Should return empty strings for all predictions
        predictions = gliner_method.predict([PredictionSample.from_text("March 10, 2024")])
        self.assertEqual([""], predictions)

        # Test with multiple prediction samples
        multiple_predictions = gliner_method.predict(
            [PredictionSample.from_text("March 10, 2024"), PredictionSample.from_text("April 5, 2025")]
        )
        self.assertEqual(["", ""], multiple_predictions)
