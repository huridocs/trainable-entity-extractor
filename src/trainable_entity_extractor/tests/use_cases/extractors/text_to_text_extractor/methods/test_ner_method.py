import unittest
from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.NerFirstAppearanceMethod import (
    NerFirstAppearanceMethod,
)

extraction_identifier = ExtractionIdentifier(run_name="unit_test", extraction_name="ner_test")


class TestNerMethod(TestCase):
    @unittest.SkipTest
    def test_ner(self):
        sample = TrainingSample(
            labeled_data=LabeledData(label_text="Huridocs", language_iso="en"),
            segment_selector_texts=["This repository belongs to Huridocs"],
        )

        extraction_data = ExtractionData(samples=[sample], extraction_identifier=extraction_identifier)
        ner_method = NerFirstAppearanceMethod(extraction_identifier)

        ner_method.train(extraction_data)

        predictions = ner_method.predict([PredictionSample.from_text("Referencing the Human Rights Council")])
        self.assertEqual(["the Human Rights Council"], predictions)

    @unittest.SkipTest
    def test_not_found_tag(self):
        sample = TrainingSample(
            labeled_data=LabeledData(label_text="Huridocs", language_iso="en"),
            segment_selector_texts=["This repository belongs to me"],
        )

        extraction_data = ExtractionData(samples=[sample], extraction_identifier=extraction_identifier)
        ner_method = NerFirstAppearanceMethod(extraction_identifier)

        ner_method.train(extraction_data)

        predictions = ner_method.predict([PredictionSample.from_text("Referencing the Human Rights Council")])
        self.assertEqual([""], predictions)

    @unittest.SkipTest
    def test_different_case(self):
        sample = TrainingSample(
            labeled_data=LabeledData(label_text="Human Rights Council", language_iso="en"),
            segment_selector_texts=["This repository belongs the human rights council"],
        )

        extraction_data = ExtractionData(samples=[sample], extraction_identifier=extraction_identifier)
        ner_method = NerFirstAppearanceMethod(extraction_identifier)

        ner_method.train(extraction_data)

        predictions = ner_method.predict([PredictionSample.from_text("This project has been build by Huridocs")])
        self.assertEqual(["Huridocs"], predictions)
