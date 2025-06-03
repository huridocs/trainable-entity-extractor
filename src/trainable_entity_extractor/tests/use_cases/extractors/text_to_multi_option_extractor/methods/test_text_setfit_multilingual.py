import unittest
from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.TextSetFitMultilingual import (
    TextSetFitMultilingual,
)

extraction_identifier = ExtractionIdentifier(run_name="setfit_multilingual", extraction_name="setfit_multilingual")


class TestTextSetFitMultilingual(TestCase):
    @unittest.SkipTest
    def test_predict_multi_value(self):
        text = "foo var Democratic Republic of the Congo Democratic People's Republic of Korea"
        options = [
            Option(id="1", label="Democratic Republic of the Congo"),
            Option(id="2", label="Democratic People's Republic of Korea"),
            Option(id="3", label="France"),
            Option(id="4", label="Congo"),
            Option(id="5", label="Korea"),
        ]
        labeled_data = LabeledData(source_text=text, values=[options[0], options[1]])
        samples = [TrainingSample(labeled_data=labeled_data)]

        text = "foo var France Congo"

        labeled_data = LabeledData(source_text=text, values=[options[2], options[3]])
        samples += [TrainingSample(labeled_data=labeled_data)]

        text = "foo var Korea"

        labeled_data = LabeledData(source_text=text, values=[options[4]])
        samples += [TrainingSample(labeled_data=labeled_data)]

        text_to_multioption = TextSetFitMultilingual(extraction_identifier, options, True)
        text_to_multioption.train(ExtractionData(samples=samples * 10, options=options))

        text = "foo var Democratic Republic of the Congo Democratic People's Republic of Korea"
        expected_options = [[options[0], options[1]]]

        prediction_sample = PredictionSample(source_text=text)
        self.assertEqual(expected_options, text_to_multioption.predict([prediction_sample]))
