import unittest
from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.NerFirstAppearanceMethod import (
    NerFirstAppearanceMethod,
)
from trainable_entity_extractor.adapters.ExtractorLogger import ExtractorLogger


class TestNerMethod(TestCase):
    def setUp(self):
        self.extraction_identifier = ExtractionIdentifier(run_name="unit_test", extraction_name="ner_test")

    def test_ner(self):
        sample = TrainingSample(
            labeled_data=LabeledData(
                label_text="Huridocs", language_iso="en", source_text="This repository belongs to Huridocs"
            )
        )

        extraction_data = ExtractionData(samples=[sample], extraction_identifier=self.extraction_identifier)
        ner_method = NerFirstAppearanceMethod(self.extraction_identifier)

        ner_method.train(extraction_data)

        prediction_data = PredictionSamplesData(
            prediction_samples=[PredictionSample(source_text="Referencing the Human Rights Council")],
            options=[],
            multi_value=False,
        )
        predictions = ner_method.predict(prediction_data)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 1)

    def test_not_found_tag(self):
        sample = TrainingSample(
            labeled_data=LabeledData(label_text="Huridocs", language_iso="en", source_text="This repository belongs to me")
        )

        extraction_data = ExtractionData(samples=[sample], extraction_identifier=self.extraction_identifier)
        ner_method = NerFirstAppearanceMethod(self.extraction_identifier)

        ner_method.train(extraction_data)

        prediction_data = PredictionSamplesData(
            prediction_samples=[PredictionSample(source_text="Referencing the Human Rights Council")],
            options=[],
            multi_value=False,
        )
        predictions = ner_method.predict(prediction_data)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 1)

    def test_different_case(self):
        sample = TrainingSample(
            labeled_data=LabeledData(
                label_text="Human Rights Council",
                language_iso="en",
                source_text="This repository belongs the human rights council",
            )
        )

        extraction_data = ExtractionData(samples=[sample], extraction_identifier=self.extraction_identifier)
        ner_method = NerFirstAppearanceMethod(self.extraction_identifier)

        ner_method.train(extraction_data)

        prediction_data = PredictionSamplesData(
            prediction_samples=[PredictionSample(source_text="Referencing the Human Rights Council")],
            options=[],
            multi_value=False,
        )
        predictions = ner_method.predict(prediction_data)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 1)

    def test_method_initialization(self):
        """Test that NerFirstAppearanceMethod can be properly initialized with real instances"""
        ner_method = NerFirstAppearanceMethod(self.extraction_identifier)

        self.assertIsNotNone(ner_method)
        self.assertEqual(ner_method.extraction_identifier, self.extraction_identifier)
