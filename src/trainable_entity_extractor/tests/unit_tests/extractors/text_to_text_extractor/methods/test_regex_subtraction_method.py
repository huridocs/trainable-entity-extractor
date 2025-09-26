from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.RegexSubtractionMethod import (
    RegexSubtractionMethod,
)
from trainable_entity_extractor.adapters.ExtractorLogger import ExtractorLogger


class TestRegexSubtractionMethod(TestCase):
    def setUp(self):
        self.extraction_identifier = ExtractionIdentifier(run_name="unit_test", extraction_name="test")

    def test_performance_text_in_front(self):
        sample_1 = TrainingSample(labeled_data=LabeledData(label_text="two", language_iso="en", source_text="one two"))
        sample_2 = TrainingSample(labeled_data=LabeledData(label_text="three", language_iso="en", source_text="one three"))

        extraction_data = ExtractionData(samples=[sample_1, sample_2], extraction_identifier=self.extraction_identifier)

        regex_method = RegexSubtractionMethod(self.extraction_identifier)

        performance = regex_method.get_performance(extraction_data, extraction_data)
        self.assertIsInstance(performance, (int, float))
        self.assertGreaterEqual(performance, 0)

    def test_performance_text_in_back(self):
        sample_1 = TrainingSample(labeled_data=LabeledData(label_text="two", language_iso="en", source_text="two other"))
        sample_2 = TrainingSample(labeled_data=LabeledData(label_text="three", language_iso="en", source_text="three other"))

        extraction_data = ExtractionData(samples=[sample_1, sample_2], extraction_identifier=self.extraction_identifier)

        regex_method = RegexSubtractionMethod(self.extraction_identifier)

        performance = regex_method.get_performance(extraction_data, extraction_data)
        self.assertIsInstance(performance, (int, float))
        self.assertGreaterEqual(performance, 0)

    def test_performance(self):
        text = "Angola, Argentina, Austria, Benin, Botswana, Brazil, Burkina Faso, Chile, Congo, Costa Rica, "
        text += "Côte d'Ivoire, Czech Republic, Ecuador, Estonia, Ethiopia, Gabon, Germany, Guatemala, India, "
        text += "Indonesia, Ireland, Italy, Japan, Kazakhstan, Kenya, Libya, Malaysia, Maldives, "
        text += "Montenegro, Pakistan, Peru, Philippines, Poland, Republic of Korea, Republic of Moldova, "
        text += "Romania, Sierra Leone, Spain, Switzerland, Thailand, Uganda, Venezuela (Bolivarian Republic of)"

        samples = [
            TrainingSample(labeled_data=LabeledData(label_text="Angola", source_text=text)),
            TrainingSample(labeled_data=LabeledData(label_text="Argentina", source_text=text)),
            TrainingSample(labeled_data=LabeledData(label_text="Austria", source_text=text)),
        ]

        extraction_data = ExtractionData(samples=samples, extraction_identifier=self.extraction_identifier)
        regex_method = RegexSubtractionMethod(self.extraction_identifier)

        performance = regex_method.get_performance(extraction_data, extraction_data)
        self.assertIsInstance(performance, (int, float))
        self.assertGreaterEqual(performance, 0)
