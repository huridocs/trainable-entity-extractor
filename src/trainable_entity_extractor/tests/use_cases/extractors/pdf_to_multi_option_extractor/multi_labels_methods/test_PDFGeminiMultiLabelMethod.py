import unittest
from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.multi_labels_methods.PDFGeminiMultiLabelMethod import (
    PDFGeminiMultiLabelMethod,
)


class TestPDFGeminiMultiLabelMethod(TestCase):
    TENANT = "unit_test"
    extraction_id = "pdf_gemini_multi_label_method"

    @unittest.skip
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
        pdf_gemini_multi_label = PDFGeminiMultiLabelMethod(extraction_identifier, options, True, __name__)

        pdf_gemini_multi_label.train(extraction_data)

        prediction_sample_1 = TrainingSample(pdf_data=pdf_data_1)
        prediction_sample_2 = TrainingSample(pdf_data=pdf_data_2)
        prediction_sample_4 = TrainingSample(pdf_data=pdf_data_4)
        prediction_samples = [prediction_sample_1, prediction_sample_2, prediction_sample_4]

        prediction_data = ExtractionData(
            multi_value=True, options=options, samples=prediction_samples, extraction_identifier=extraction_identifier
        )
        predictions = pdf_gemini_multi_label.predict(prediction_data)

        self.assertEqual(3, len(predictions))
        self.assertIn(Value(id="1", label="1"), predictions[0])
        self.assertIn(Value(id="2", label="2"), predictions[1])
        self.assertIn(Value(id="3", label="3"), predictions[1])
        self.assertIn(Value(id="4", label="4"), predictions[2])
        self.assertIn(Value(id="1", label="1"), predictions[2])
        self.assertNotIn(Value(id="5", label="5"), predictions[0])
        self.assertNotIn(Value(id="4", label="4"), predictions[1])
        self.assertNotIn(Value(id="3", label="3"), predictions[2])
