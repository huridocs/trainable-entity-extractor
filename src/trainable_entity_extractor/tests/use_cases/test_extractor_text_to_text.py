import shutil
from unittest import TestCase

from trainable_entity_extractor.use_cases.TrainableEntityExtractor import TrainableEntityExtractor
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.TrainingSample import TrainingSample

extraction_id = "test_extractor_text_to_text"
extraction_identifier = ExtractionIdentifier(extraction_name=extraction_id)


class TestExtractorTextToText(TestCase):
    def setUp(self):
        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)

    def test_predictions_same_input_output(self):
        labeled_data = LabeledData(label_text="one", language_iso="en")
        sample = [TrainingSample(labeled_data=labeled_data, segment_selector_texts=["two"])]
        extraction_data = ExtractionData(samples=sample, extraction_identifier=extraction_identifier)

        trainable_entity_extractor = TrainableEntityExtractor(extraction_identifier=extraction_identifier)
        trainable_entity_extractor.train(extraction_data)

        texts = ["test 0", "test 1", "test 2"]
        predictions_samples = [PredictionSample.from_text(text, str(i)) for i, text in enumerate(texts)]
        suggestions = trainable_entity_extractor.predict(predictions_samples)

        self.assertEqual(3, len(suggestions))
        self.assertEqual("default", suggestions[0].tenant)
        self.assertEqual(extraction_id, suggestions[0].id)
        self.assertEqual("0", suggestions[0].entity_name)
        self.assertEqual("test 0", suggestions[0].text)
        self.assertEqual("1", suggestions[1].entity_name)
        self.assertEqual("test 1", suggestions[1].text)
        self.assertEqual("2", suggestions[2].entity_name)
        self.assertEqual("test 2", suggestions[2].text)

    def test_predictions_from_source_text_in_labeled_data(self):
        samples = [TrainingSample(labeled_data=LabeledData(label_text="1", language_iso="en", source_text="foo 1"))]
        samples += [TrainingSample(labeled_data=LabeledData(label_text="2", language_iso="en", source_text="2 var"))]
        extraction_data = ExtractionData(samples=samples, extraction_identifier=extraction_identifier)

        trainable_entity_extractor = TrainableEntityExtractor(extraction_identifier=extraction_identifier)
        trainable_entity_extractor.train(extraction_data)

        texts = ["test 0"]
        predictions_samples = [PredictionSample.from_text(text, str(i)) for i, text in enumerate(texts)]
        predictions_samples[0].segment_selector_texts = []
        suggestions = trainable_entity_extractor.predict(predictions_samples)

        self.assertEqual(1, len(suggestions))
        self.assertEqual("0", suggestions[0].text)
