import unittest
from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_labels_methods.PDFOllamaMultiLabelMethod import (
    PDFOllamaMultiLabelMethod,
)


class TestPDFOllamaMultiLabelMethod(TestCase):
    TENANT = "unit_test"
    extraction_id = "pdf_ollama_multi_label_method"

    @unittest.SkipTest
    def test_train_and_predict(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [
            Option(id="1", label="1"),
            Option(id="2", label="2"),
            Option(id="3", label="3"),
            Option(id="4", label="4"),
            Option(id="5", label="5"),
        ]

        pdf_data_1 = PdfData.from_texts(["point 1"])
        pdf_data_2 = PdfData.from_texts(["point 2"])
        pdf_data_4 = PdfData.from_texts(["point 4"])

        samples = [
            TrainingSample(pdf_data=pdf_data_1, labeled_data=LabeledData(values=[options[0]])),
            TrainingSample(pdf_data=pdf_data_2, labeled_data=LabeledData(values=[options[1], options[2]])),
            TrainingSample(pdf_data=pdf_data_4, labeled_data=LabeledData(values=[options[3], options[0]])),
        ]

        extraction_data = ExtractionData(
            multi_value=True, options=options, samples=samples, extraction_identifier=extraction_identifier
        )
        pdf_ollama_multi_label = PDFOllamaMultiLabelMethod(extraction_identifier)

        pdf_ollama_multi_label.train(extraction_data)

        prediction_samples = [
            PredictionSample.from_pdf_data(pdf_data_1),
            PredictionSample.from_pdf_data(pdf_data_2),
            PredictionSample.from_pdf_data(pdf_data_4),
        ]

        prediction_samples_data = PredictionSamplesData(
            prediction_samples=prediction_samples,
            options=options,
            multi_value=True,
        )
        predictions = pdf_ollama_multi_label.predict(prediction_samples_data)

        self.assertEqual(3, len(predictions))
        self.assertIn(Value(id="1", label="1"), predictions[0])
        self.assertIn(Value(id="2", label="2"), predictions[1])
        self.assertIn(Value(id="3", label="3"), predictions[1])
        self.assertIn(Value(id="4", label="4"), predictions[2])
        self.assertIn(Value(id="1", label="1"), predictions[2])
        self.assertNotIn(Value(id="5", label="5"), predictions[0])
        self.assertNotIn(Value(id="4", label="4"), predictions[1])
        self.assertNotIn(Value(id="3", label="3"), predictions[2])
