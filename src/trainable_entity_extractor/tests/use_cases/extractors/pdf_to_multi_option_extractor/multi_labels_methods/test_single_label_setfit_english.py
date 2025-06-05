import shutil
import unittest
from os.path import join
from unittest import TestCase

import torch

from trainable_entity_extractor.config import DATA_PATH
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.multi_labels_methods.SingleLabelSetFitEnglishMethod import (
    SingleLabelSetFitEnglishMethod,
)


class TestSetFitSingleLabelEnglishMethod(TestCase):
    TENANT = "unit_test"
    extraction_id = "multi_option_extraction_test"

    def tearDown(self):
        shutil.rmtree(join(DATA_PATH, self.TENANT), ignore_errors=True)

    @unittest.SkipTest
    def test_train_and_predict(self):
        if not torch.cuda.is_available():
            return
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="0", label="0"), Option(id="1", label="1"), Option(id="2", label="2")]

        pdf_data_1 = PdfData.from_texts(["point 1"])
        pdf_data_2 = PdfData.from_texts(["point 2"])
        pdf_data_3 = PdfData.from_texts(["point 3"])

        samples = [
            TrainingSample(pdf_data=pdf_data_1, labeled_data=LabeledData(values=[options[0]])),
            TrainingSample(pdf_data=pdf_data_2, labeled_data=LabeledData(values=[options[1]])),
            TrainingSample(pdf_data=pdf_data_3, labeled_data=LabeledData(values=[options[2]])),
        ]

        extraction_data = ExtractionData(
            multi_value=False, options=options, samples=samples, extraction_identifier=extraction_identifier
        )
        setfit_english_method = SingleLabelSetFitEnglishMethod(extraction_identifier, options, False)

        try:
            setfit_english_method.train(extraction_data)
        except Exception as e:
            self.fail(f"train() raised {type(e).__name__}")

        prediction_sample_1 = TrainingSample(pdf_data=pdf_data_1)
        prediction_sample_2 = TrainingSample(pdf_data=pdf_data_2)
        prediction_sample_3 = TrainingSample(pdf_data=pdf_data_3)
        prediction_samples = [prediction_sample_1, prediction_sample_2, prediction_sample_3]

        prediction_data = ExtractionData(
            multi_value=False, options=options, samples=prediction_samples, extraction_identifier=extraction_identifier
        )
        predictions = setfit_english_method.predict(prediction_data)

        self.assertEqual(3, len(predictions))
        self.assertIn(Option(id="0", label="0"), predictions[0])
        self.assertIn(Option(id="1", label="1"), predictions[1])
        self.assertNotIn(Option(id="5", label="5"), predictions[0])
        self.assertNotIn(Option(id="4", label="4"), predictions[1])
