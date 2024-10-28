import shutil
from os.path import join
from unittest import TestCase

from trainable_entity_extractor.config import DATA_PATH
from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.LabeledData import LabeledData
from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.data.TrainingSample import TrainingSample
from trainable_entity_extractor.extractors.text_to_text_extractor.TextToTextExtractor import TextToTextExtractor

tenant = "semantic_tenant_to_be_removed"
extraction_id = "extraction_id"

extraction_identifier = ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id)


class TestTextToTextExtractor(TestCase):

    def setUp(self):
        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

    def test_predictions_same_input_output(self):
        sample = [TrainingSample(labeled_data=LabeledData(label_text="one", language_iso="en"), tags_texts=["two"])]
        extraction_data = ExtractionData(samples=sample, extraction_identifier=extraction_identifier)

        text_to_text_extractor = TextToTextExtractor(extraction_identifier=extraction_identifier)
        text_to_text_extractor.can_be_used(extraction_data)
        texts = ["test 0", "test 1", "test 2"]
        predictions_samples = [PredictionSample.from_text(text, str(i)) for i, text in enumerate(texts)]
        suggestions = text_to_text_extractor.get_suggestions(predictions_samples)

        self.assertEqual(3, len(suggestions))
        self.assertEqual(tenant, suggestions[0].tenant)
        self.assertEqual(extraction_id, suggestions[0].id)
        self.assertEqual("0", suggestions[0].entity_name)
        self.assertEqual("test 0", suggestions[0].text)
        self.assertEqual("1", suggestions[1].entity_name)
        self.assertEqual("test 1", suggestions[1].text)
        self.assertEqual("2", suggestions[2].entity_name)
        self.assertEqual("test 2", suggestions[2].text)

    def test_predictions_two_samples(self):
        sample_1 = [TrainingSample(labeled_data=LabeledData(label_text="one", language_iso="en"), tags_texts=["one two"])]
        sample_2 = [TrainingSample(labeled_data=LabeledData(label_text="one", language_iso="en"), tags_texts=["one two"])]
        extraction_data = ExtractionData(samples=sample_1 + sample_2, extraction_identifier=extraction_identifier)

        text_to_text_extractor = TextToTextExtractor(extraction_identifier=extraction_identifier)
        text_to_text_extractor.create_model(extraction_data)

        suggestions = text_to_text_extractor.get_suggestions([PredictionSample.from_text("one two", "entity_name")])

        self.assertEqual(1, len(suggestions))
        self.assertEqual(tenant, suggestions[0].tenant)
        self.assertEqual(extraction_id, suggestions[0].id)
        self.assertEqual("entity_name", suggestions[0].entity_name)
        self.assertEqual("one", suggestions[0].text)

    def test_predictions_input_without_spaces(self):
        sample = [
            TrainingSample(
                labeled_data=LabeledData(label_text="onetwothree", language_iso="en"), tags_texts=["one two", "three"]
            )
        ]
        extraction_data = ExtractionData(samples=sample * 3, extraction_identifier=extraction_identifier)

        text_to_text_extractor = TextToTextExtractor(extraction_identifier=extraction_identifier)
        text_to_text_extractor.create_model(extraction_data)

        suggestions = text_to_text_extractor.get_suggestions(
            [PredictionSample.from_text("one two three four", "entity_name")]
        )

        self.assertEqual(1, len(suggestions))
        self.assertEqual(tenant, suggestions[0].tenant)
        self.assertEqual(extraction_id, suggestions[0].id)
        self.assertEqual("entity_name", suggestions[0].entity_name)
        self.assertEqual("onetwothreefour", suggestions[0].text)
