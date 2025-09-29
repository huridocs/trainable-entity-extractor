import shutil
import unittest
from os.path import join
from time import time
from unittest import TestCase

import torch

from trainable_entity_extractor.config import DATA_PATH
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.MT5TrueCaseEnglishSpanishMethod import (
    MT5TrueCaseEnglishSpanishMethod,
)
from trainable_entity_extractor.adapters.ExtractorLogger import ExtractorLogger


class TestMT5(TestCase):
    def setUp(self):
        shutil.rmtree(join(DATA_PATH, "test"), ignore_errors=True)
        self.extraction_identifier = ExtractionIdentifier(run_name="unit_test", extraction_name="test")

    def tearDown(self):
        shutil.rmtree(join(DATA_PATH, "test"), ignore_errors=True)

    @unittest.skip("Requires GPU and substantial compute time")
    def test_train(self):
        start = time()
        print("GPU available?")
        print(torch.cuda.is_available())

        samples_1 = [TrainingSample(labeled_data=LabeledData(label_text="foo", source_text="1/ foo end"))] * 5
        samples_2 = [TrainingSample(labeled_data=LabeledData(label_text="var", source_text="2/ var end"))] * 5

        extraction_data = ExtractionData(samples=samples_1 + samples_2, extraction_identifier=self.extraction_identifier)
        mt5_true_case_english_spanish = MT5TrueCaseEnglishSpanishMethod(self.extraction_identifier)

        mt5_true_case_english_spanish.train(extraction_data)

        prediction_data = PredictionSamplesData(
            prediction_samples=[PredictionSample(source_text="3/ test end")], options=[], multi_value=False
        )
        predictions = mt5_true_case_english_spanish.predict(prediction_data)

        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 1)
        print(f"Training and prediction completed in {time() - start} seconds")

    def test_method_initialization(self):
        """Test that MT5TrueCaseEnglishSpanishMethod can be properly initialized with real instances"""
        mt5_method = MT5TrueCaseEnglishSpanishMethod(self.extraction_identifier)

        self.assertIsNotNone(mt5_method)
        self.assertEqual(mt5_method.extraction_identifier, self.extraction_identifier)
