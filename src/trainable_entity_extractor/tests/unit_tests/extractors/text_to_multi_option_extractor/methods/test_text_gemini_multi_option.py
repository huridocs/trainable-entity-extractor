import unittest
from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.methods.gemini_multi_option.TextGeminiMultiOption import (
    TextGeminiMultiOption,
)


class TestTextGeminiMultiOption(TestCase):
    @unittest.skip("Requires real Gemini API key")
    def test_text_gemini_multi_option(self):
        extraction_identifier = ExtractionIdentifier(run_name="unit_test", extraction_name="gemini_multi_option_countries")

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

        text_to_multioption = TextGeminiMultiOption(extraction_identifier, self.__class__.__name__)
        extraction_data = ExtractionData(samples=samples * 10, options=options)
        text_to_multioption.options = options
        text_to_multioption.multi_value = extraction_data.multi_value
        text_to_multioption.train(extraction_data)

        text = "foo var Democratic Republic of the Congo Democratic People's Republic of Korea"
        expected_options = [[options[0], options[1]]]

        prediction_samples_data = PredictionSamplesData(
            prediction_samples=[PredictionSample(source_text=text)], options=options, multi_value=False
        )
        predictions = text_to_multioption.predict(prediction_samples_data)

        self.assertEqual(1, len(predictions))
        self.assertEqual(set(expected_options[0]), set(predictions[0]))

    @unittest.skip("Requires real Gemini API key")
    def test_text_gemini_multi_option_second(self):
        extraction_identifier = ExtractionIdentifier(run_name="unit_test", extraction_name="gemini_multi_option_fruits")

        options = [
            Option(id="1", label="apple"),
            Option(id="2", label="banana"),
            Option(id="3", label="cherry"),
            Option(id="4", label="orange"),
            Option(id="5", label="grape"),
        ]

        text1 = "I like apple and banana"
        samples = [TrainingSample(labeled_data=LabeledData(source_text=text1, values=[options[0], options[1]]))]
        text2 = "banana cherry banana"
        samples.append(TrainingSample(labeled_data=LabeledData(source_text=text2, values=[options[1], options[2]])))
        text3 = "orange"
        samples.append(TrainingSample(labeled_data=LabeledData(source_text=text3, values=[options[3]])))

        text_to_multioption = TextGeminiMultiOption(extraction_identifier, self.__class__.__name__)
        extraction_data = ExtractionData(samples=samples * 10, options=options)
        text_to_multioption.options = options
        text_to_multioption.multi_value = extraction_data.multi_value
        text_to_multioption.train(extraction_data)

        test_text = "I ate apple, cherry, and orange"
        expected = [[options[0], options[2], options[3]]]

        prediction_samples_data = PredictionSamplesData(
            prediction_samples=[PredictionSample(source_text=test_text)], options=options, multi_value=False
        )
        predictions = text_to_multioption.predict(prediction_samples_data)

        self.assertEqual(1, len(predictions))
        self.assertEqual(set(expected[0]), set(predictions[0]))
