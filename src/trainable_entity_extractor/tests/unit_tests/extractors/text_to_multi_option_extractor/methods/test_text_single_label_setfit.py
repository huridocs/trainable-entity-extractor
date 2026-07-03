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
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.methods.TextSingleLabelSetFit import (
    TextSingleLabelSetFit,
)


class TestTextSingleLabelSetFit(TestCase):
    TENANT = "unit_test"
    extraction_id = "text_single_label_setfit_test"

    def tearDown(self):
        shutil.rmtree(join(DATA_PATH, self.TENANT, self.extraction_id), ignore_errors=True)

    @unittest.skip("Skipping GPU test in CI/CD")
    def test_train_and_predict(self):
        if not torch.cuda.is_available():
            return

        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="fruit", label="fruit"), Option(id="animal", label="animal")]

        samples = [
            TrainingSample(labeled_data=LabeledData(values=[options[0]], source_text="I like apples")),
            TrainingSample(labeled_data=LabeledData(values=[options[0]], source_text="bananas are yellow")),
            TrainingSample(labeled_data=LabeledData(values=[options[0]], source_text="oranges are citrus")),
            TrainingSample(labeled_data=LabeledData(values=[options[0]], source_text="grapes grow in bunches")),
            TrainingSample(labeled_data=LabeledData(values=[options[1]], source_text="dogs are loyal")),
            TrainingSample(labeled_data=LabeledData(values=[options[1]], source_text="cats are independent")),
            TrainingSample(labeled_data=LabeledData(values=[options[1]], source_text="birds can fly")),
            TrainingSample(labeled_data=LabeledData(values=[options[1]], source_text="fish swim in water")),
        ]

        extraction_data = ExtractionData(
            multi_value=False, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        method = TextSingleLabelSetFit(extraction_identifier)

        try:
            method.train(extraction_data)
        except Exception as e:
            self.fail(f"train() raised {type(e).__name__}: {e}")

        prediction_samples = [
            PredictionSample(source_text="I love strawberries"),
            PredictionSample(source_text="rabbits are cute"),
        ]
        prediction_data = PredictionSamplesData(multi_value=False, options=options, prediction_samples=prediction_samples)
        predictions = method.predict(prediction_data)

        self.assertEqual(2, len(predictions))
        # "strawberries" → fruit
        self.assertEqual(1, len(predictions[0]))
        self.assertEqual("fruit", predictions[0][0].label)
        # "rabbits" → animal
        self.assertEqual(1, len(predictions[1]))
        self.assertEqual("animal", predictions[1][0].label)
        # Verify actual Option objects from the original options list
        self.assertIs(options[0], predictions[0][0])
        self.assertIs(options[1], predictions[1][0])
