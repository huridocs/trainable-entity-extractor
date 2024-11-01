import shutil
import unittest
from os.path import join
from unittest import TestCase

import torch

from trainable_entity_extractor.config import DATA_PATH
from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.LabeledData import LabeledData
from trainable_entity_extractor.data.Option import Option
from trainable_entity_extractor.data.PdfData import PdfData
from trainable_entity_extractor.data.TrainingSample import TrainingSample
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_labels_methods.SetFitMultilingualMethod import (
    SetFitMultilingualMethod,
)


class TestSetFitMultilingualMethod(TestCase):
    TENANT = "setfit_multilingual_test"
    extraction_id = "extraction_id"

    def tearDown(self):
        shutil.rmtree(join(DATA_PATH, self.TENANT), ignore_errors=True)

    @unittest.SkipTest
    def test_train_and_predict(self):
        if not torch.cuda.is_available():
            return
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [
            Option(id="1", label="1"),
            Option(id="2", label="2"),
            Option(id="3", label="3"),
            Option(id="4", label="4"),
            Option(id="5", label="5"),
        ]

        pdf_data_1 = PdfData.from_texts(["point 1"])
        pdf_data_2 = PdfData.from_texts(["point 2 point 3"])
        pdf_data_3 = PdfData.from_texts(["point 3"])
        pdf_data_4 = PdfData.from_texts(["point 4 point 1"])
        pdf_data_5 = PdfData.from_texts(["point 5"])

        samples = [
            TrainingSample(pdf_data_1, LabeledData(values=[options[0]])),
            TrainingSample(pdf_data_2, LabeledData(values=[options[1], options[2]])),
            TrainingSample(pdf_data_3, LabeledData(values=[options[2]])),
            TrainingSample(pdf_data_4, LabeledData(values=[options[3], options[0]])),
            TrainingSample(pdf_data_5, LabeledData(values=[options[4]])),
        ]

        extraction_data = ExtractionData(
            multi_value=True, options=options, samples=samples, extraction_identifier=extraction_identifier
        )
        setfit_multilingual_method = SetFitMultilingualMethod(extraction_identifier, options, True)

        try:
            setfit_multilingual_method.train(extraction_data)
        except Exception as e:
            self.fail(f"train() raised {type(e).__name__}")

        prediction_sample_1 = TrainingSample(pdf_data=pdf_data_1)
        prediction_sample_2 = TrainingSample(pdf_data=pdf_data_2)
        prediction_sample_4 = TrainingSample(pdf_data=pdf_data_4)
        prediction_samples = [prediction_sample_1, prediction_sample_2, prediction_sample_4]

        prediction_data = ExtractionData(
            multi_value=True, options=options, samples=prediction_samples, extraction_identifier=extraction_identifier
        )
        predictions = setfit_multilingual_method.predict(prediction_data)

        self.assertEqual(3, len(predictions))
        self.assertIn(Option(id="1", label="1"), predictions[0])
        self.assertNotIn(Option(id="5", label="5"), predictions[0])

        self.assertIn(Option(id="2", label="2"), predictions[1])
        self.assertIn(Option(id="3", label="3"), predictions[1])
        self.assertNotIn(Option(id="4", label="4"), predictions[1])

        self.assertIn(Option(id="4", label="4"), predictions[2])
        self.assertIn(Option(id="1", label="1"), predictions[2])
        self.assertNotIn(Option(id="3", label="3"), predictions[2])
