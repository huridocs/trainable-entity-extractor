import shutil
from unittest import TestCase

from trainable_entity_extractor.TrainableEntityExtractor import TrainableEntityExtractor
from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.LabeledData import LabeledData
from trainable_entity_extractor.data.Option import Option
from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.data.TrainingSample import TrainingSample

extraction_id = "test_extractor_text_to_multi_option"
extraction_identifier = ExtractionIdentifier(extraction_name=extraction_id)


class TestExtractorTextToMultiOption(TestCase):
    def setUp(self):
        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)

    def test_get_text_multi_option_suggestions(self):
        options = [Option(id="1", label="abc"), Option(id="2", label="dfg"), Option(id="3", label="hij")]

        values_1 = [Option(id="1", label="abc"), Option(id="2", label="dfg")]
        labeled_data_1 = LabeledData(language_iso="en", values=values_1, source_text="abc dfg")

        values_2 = [Option(id="2", label="dfg"), Option(id="3", label="hij")]
        labeled_data_2 = LabeledData(language_iso="en", values=values_2, source_text="dfg hij")

        sample = [TrainingSample(labeled_data=labeled_data_1), TrainingSample(labeled_data=labeled_data_2)]
        extraction_data = ExtractionData(
            samples=sample, extraction_identifier=extraction_identifier, multi_value=True, options=options
        )

        trainable_entity_extractor = TrainableEntityExtractor(extraction_identifier=extraction_identifier)
        trainable_entity_extractor.train(extraction_data)

        suggestions = trainable_entity_extractor.predict([PredictionSample.from_text("foo var dfg hij foo var")])

        self.assertEqual(1, len(suggestions))
        self.assertEqual(extraction_id, suggestions[0].id)
        self.assertEqual(options[1:], suggestions[0].values)
