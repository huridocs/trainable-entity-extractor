from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.TextToCountries import (
    TextToCountries,
)

extraction_identifier = ExtractionIdentifier(run_name="text_to_countries", extraction_name="text_to_countries")


class TestTextToCountries(TestCase):
    def test_can_be_used(self):
        text_to_countries = TextToCountries(extraction_identifier, [], False)
        self.assertFalse(text_to_countries.can_be_used(ExtractionData(samples=[], options=[Option(id="1", label="1")])))
        self.assertTrue(text_to_countries.can_be_used(ExtractionData(samples=[], options=[Option(id="1", label="Spain")])))
        self.assertFalse(
            text_to_countries.can_be_used(
                ExtractionData(samples=[], options=[Option(id="1", label="Spain"), Option(id="2", label="foo")])
            )
        )
        self.assertTrue(
            text_to_countries.can_be_used(
                ExtractionData(samples=[], options=[Option(id="1", label="Spain"), Option(id="2", label="Chile")])
            )
        )

    def test_predict_single_value(self):
        text = "foo var chile spain france"
        sample = PredictionSample(source_text=text)
        options = [Option(id="1", label="Spain"), Option(id="2", label="Chile"), Option(id="3", label="France")]
        text_to_countries = TextToCountries(extraction_identifier, options, False)
        text_to_countries.train(ExtractionData(samples=[TrainingSample()], options=options))
        self.assertEqual([[Option(id="2", label="Chile")]], text_to_countries.predict([sample]))

    def test_predict_multi_value(self):
        text = "foo var Democratic Republic of the Congo Democratic People's Republic of Korea"
        sample = PredictionSample(source_text=text)
        options = [
            Option(id="1", label="Democratic Republic of the Congo"),
            Option(id="2", label="Democratic People's Republic of Korea"),
            Option(id="3", label="France"),
            Option(id="4", label="Congo"),
            Option(id="5", label="Korea"),
        ]
        text_to_countries = TextToCountries(extraction_identifier, options, True)
        text_to_countries.train(ExtractionData(samples=[TrainingSample()], options=options))
        expected_options = [[options[0], options[1]]]
        self.assertEqual(expected_options, text_to_countries.predict([sample]))

    def test_predict_non_country_name(self):
        text = "foo text more text Cote d'Ivoire and var"
        sample = PredictionSample(source_text=text)
        options = [Option(id="1", label="Spain"), Option(id="2", label="Chile"), Option(id="3", label="Côte d'Ivoire")]
        text_to_countries = TextToCountries(extraction_identifier, options, False)
        text_to_countries.train(ExtractionData(samples=[TrainingSample()], options=options))
        self.assertEqual([[Option(id="3", label="Côte d'Ivoire")]], text_to_countries.predict([sample]))
