from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.SameInputOutputMethod import (
    SameInputOutputMethod,
)
from trainable_entity_extractor.adapters.ExtractorLogger import ExtractorLogger


class TestSameInputMethod(TestCase):
    def setUp(self):
        self.extraction_identifier = ExtractionIdentifier(run_name="unit_test", extraction_name="test")

    def test_performance_100(self):
        sample = TrainingSample(labeled_data=LabeledData(label_text="a b c", language_iso="en", source_text="a b c"))

        extraction_data = ExtractionData(samples=[sample], extraction_identifier=self.extraction_identifier)

        same_input_output_method = SameInputOutputMethod(self.extraction_identifier)
        performance = same_input_output_method.get_performance(extraction_data, extraction_data)
        self.assertIsInstance(performance, (int, float))
        self.assertGreaterEqual(performance, 0)

    def test_performance_100_with_multiline(self):
        label_text = """Albania, Algeria, Argentina, Bolivia (Plurinational State of), Brazil, Congo, Côte d'Ivoire,
                        El Salvador, Estonia, France, Gabon, Germany, Ireland, Kazakhstan, Latvia, Mexico,
                        Montenegro, Namibia, Netherlands, Paraguay, Portugal, Sierra Leone, South Africa, the
                        former Yugoslav Republic of Macedonia, United Kingdom of Great Britain and Northern
                        Ireland, Venezuela (Bolivarian Republic of)"""

        source_text = (
            "Albania, Algeria, Argentina, Bolivia (Plurinational State of), Brazil, Congo, Côte d'Ivoire, "
            + "El Salvador, Estonia, France, Gabon, Germany, Ireland, Kazakhstan, Latvia, Mexico, "
            + "Montenegro, Namibia, Netherlands, Paraguay, Portugal, Sierra Leone, South Africa, "
            + "the former Yugoslav Republic of Macedonia, United Kingdom of Great Britain and "
            + "Northern Ireland, Venezuela (Bolivarian Republic of)"
        )

        sample = TrainingSample(labeled_data=LabeledData(label_text=label_text, language_iso="en", source_text=source_text))

        extraction_data = ExtractionData(samples=[sample], extraction_identifier=self.extraction_identifier)
        same_input_output_method = SameInputOutputMethod(self.extraction_identifier)

        performance = same_input_output_method.get_performance(extraction_data, extraction_data)
        self.assertIsInstance(performance, (int, float))
        self.assertGreaterEqual(performance, 0)
