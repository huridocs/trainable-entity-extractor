import shutil
from os.path import join
from time import time
from unittest import TestCase

import torch

from trainable_entity_extractor.config import DATA_PATH
from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.LabeledData import LabeledData
from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.data.TrainingSample import TrainingSample
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.MT5TrueCaseEnglishSpanishMethod import (
    MT5TrueCaseEnglishSpanishMethod,
)


class TestMT5(TestCase):
    def setUp(self):
        shutil.rmtree(join(DATA_PATH, "test"), ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(join(DATA_PATH, "test"), ignore_errors=True)

    def test_train(self):
        start = time()
        print("GPU available?")
        print(torch.cuda.is_available())
        extraction_identifier = ExtractionIdentifier(run_name="test", extraction_name="test")

        samples_1 = [TrainingSample(labeled_data=LabeledData(label_text="foo"), segment_selector_texts=["1/ foo end"])] * 5
        samples_2 = [TrainingSample(labeled_data=LabeledData(label_text="var"), segment_selector_texts=["2/ var end"])] * 5

        extraction_data = ExtractionData(samples=samples_1 + samples_2, extraction_identifier=extraction_identifier)
        mt5_true_case_english_spanish = MT5TrueCaseEnglishSpanishMethod(extraction_identifier)

        mt5_true_case_english_spanish.train(extraction_data)

        prediction_sample = PredictionSample.from_text("1/ foo end")
        prediction_sample_empty = PredictionSample.from_text("")
        predictions = mt5_true_case_english_spanish.predict([prediction_sample] + [prediction_sample_empty])

        self.assertEqual(2, len(predictions))
        self.assertEqual("foo", predictions[0])
        self.assertEqual("", predictions[1])
        print(f"Predictions in {round(time() - start, 2)} seconds")
