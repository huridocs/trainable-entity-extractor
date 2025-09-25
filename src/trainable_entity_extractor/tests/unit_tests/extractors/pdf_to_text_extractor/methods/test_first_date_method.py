from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.adapters.extractors.pdf_to_text_extractor.methods.FirstDateMethod import FirstDateMethod

extraction_identifier = ExtractionIdentifier(run_name="unit_test", extraction_name="date_test")


class TestFirstDateMethod(TestCase):
    def test_predict_multiple_dates_spanish(self):
        text = "Informe no. 52/16 caso 12.521 fondo Maria Laura órdenes guerra y otros Chile 30 de noviembre de 2016"
        sample = TrainingSample(
            labeled_data=LabeledData(label_text="2016-11-30", language_iso="es"), segment_selector_texts=[text]
        )

        extraction_data = ExtractionData(samples=[sample for _ in range(6)], extraction_identifier=extraction_identifier)
        first_date_method = FirstDateMethod(extraction_identifier)

        first_date_method.train(extraction_data)

        prediction_samples_data = PredictionSamplesData(prediction_samples=[PredictionSample.from_text(text)])
        predictions = first_date_method.predict(prediction_samples_data)
        self.assertEqual(["2016-11-30"], predictions)
