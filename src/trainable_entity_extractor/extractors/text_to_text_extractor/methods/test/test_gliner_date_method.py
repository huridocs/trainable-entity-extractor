from unittest import TestCase

from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.LabeledData import LabeledData
from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.data.TrainingSample import TrainingSample
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.GlinerDateParserMethod import (
    GlinerDateParserMethod,
)

extraction_identifier = ExtractionIdentifier(run_name="date_test", extraction_name="date_test")


class TestGlinerDateMethod(TestCase):
    def test_predict(self):
        sample = TrainingSample(
            labeled_data=LabeledData(label_text="1982-06-05", language_iso="en"), tags_texts=["5 Jun 1982"]
        )

        extraction_data = ExtractionData(samples=[sample for _ in range(6)], extraction_identifier=extraction_identifier)
        gliner_method = GlinerDateParserMethod(extraction_identifier)

        gliner_method.train(extraction_data)

        predictions = gliner_method.predict([PredictionSample.from_text("5 Jun 1982")])
        self.assertEqual(["1982-06-05"], predictions)

    def test_predict_multiple_dates_spanish(self):
        text = "Informe no. 52/16 caso 12.521 fondo Maria Laura órdenes guerra y otros Chile 30 de noviembre de 2016"
        sample = TrainingSample(labeled_data=LabeledData(label_text="2016-11-30", language_iso="es"), tags_texts=[text])

        extraction_data = ExtractionData(samples=[sample for _ in range(6)], extraction_identifier=extraction_identifier)
        gliner_method = GlinerDateParserMethod(extraction_identifier)

        gliner_method.train(extraction_data)

        predictions = gliner_method.predict([PredictionSample.from_text(text)])
        self.assertEqual(["2016-11-30"], predictions)

    def test_performance_multiple_tags(self):
        sample = TrainingSample(
            labeled_data=LabeledData(label_text="1981-05-13", language_iso="es"), tags_texts=["13 May", "1981"]
        )

        extraction_data = ExtractionData(samples=[sample for _ in range(6)], extraction_identifier=extraction_identifier)
        gliner_method = GlinerDateParserMethod(extraction_identifier)

        self.assertEqual(100, gliner_method.performance(extraction_data, extraction_data))
