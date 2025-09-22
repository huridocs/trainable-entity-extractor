from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.RegexSubtractionMethod import (
    RegexSubtractionMethod,
)

extraction_identifier = ExtractionIdentifier(run_name="unit_test", extraction_name="test")


class TestRegexSubtractionMethod(TestCase):
    def test_performance_text_in_front(self):
        sample_1 = TrainingSample(
            labeled_data=LabeledData(label_text="two", language_iso="en"), segment_selector_texts=["one two"]
        )
        sample_2 = TrainingSample(
            labeled_data=LabeledData(label_text="three", language_iso="en"), segment_selector_texts=["one three"]
        )

        extraction_data = ExtractionData(samples=[sample_1, sample_2], extraction_identifier=extraction_identifier)

        regex_method = RegexSubtractionMethod(extraction_identifier)

        self.assertEqual(100, regex_method.get_performance(extraction_data, extraction_data))

    def test_performance_text_in_back(self):
        sample_1 = TrainingSample(
            labeled_data=LabeledData(label_text="two", language_iso="en"), segment_selector_texts=["two other"]
        )
        sample_2 = TrainingSample(
            labeled_data=LabeledData(label_text="three", language_iso="en"), segment_selector_texts=["three other"]
        )

        extraction_data = ExtractionData(samples=[sample_1, sample_2], extraction_identifier=extraction_identifier)

        regex_method = RegexSubtractionMethod(extraction_identifier)

        self.assertEqual(100, regex_method.get_performance(extraction_data, extraction_data))

    def test_performance(self):
        text = "Angola, Argentina, Austria, Benin, Botswana, Brazil, Burkina Faso, Chile, Congo, Costa Rica, "
        text += "Côte d’Ivoire, Czech Republic, Ecuador, Estonia, Ethiopia, Gabon, Germany, Guatemala, India, "
        text += "Indonesia, Ireland, Italy, Japan, Kazakhstan, Kenya, Libya, Malaysia, Maldives, "
        text += "Montenegro, Pakistan, Peru, Philippines, Poland, Republic of Korea, Republic of Moldova, "
        text += "Romania, Sierra Leone, Spain, Switzerland, Thailand, Uganda, Venezuela (Bolivarian Republic of)"

        texts = [
            "Angola, Argentina, Austria, Benin, Botswana, Brazil, Burkina Faso, Chile,",
            "Congo, Costa Rica, Côte d’Ivoire, Czech Republic, Ecuador, Estonia,",
            "Ethiopia, Gabon, Germany, Guatemala, India, Indonesia, Ireland, Italy,",
            "Japan, Kazakhstan, Kenya, Libya, Malaysia, Maldives, Montenegro,",
            "Pakistan, Peru, Philippines, Poland, Republic of Korea, Republic of",
            "Moldova, Romania, Sierra Leone, Spain, Switzerland, Thailand, Uganda,",
            "Venezuela (Bolivarian Republic of)",
        ]

        sample = TrainingSample(labeled_data=LabeledData(label_text=text, language_iso="en"), segment_selector_texts=texts)

        extraction_data = ExtractionData(samples=[sample], extraction_identifier=extraction_identifier)

        regex_method = RegexSubtractionMethod(extraction_identifier)

        self.assertEqual(100, regex_method.get_performance(extraction_data, extraction_data))

    def test_performance_text_front_and_back(self):
        sample_1 = TrainingSample(
            labeled_data=LabeledData(label_text="two", language_iso="en"), segment_selector_texts=["1/ two final"]
        )
        sample_2 = TrainingSample(
            labeled_data=LabeledData(label_text="three", language_iso="en"), segment_selector_texts=["1/ three final"]
        )

        extraction_data = ExtractionData(samples=[sample_1, sample_2], extraction_identifier=extraction_identifier)

        regex_method = RegexSubtractionMethod(extraction_identifier)

        self.assertEqual(100, regex_method.get_performance(extraction_data, extraction_data))

    def test_predict(self):
        sample_1 = TrainingSample(
            labeled_data=LabeledData(label_text="foo", language_iso="en"), segment_selector_texts=["1/ foo end"]
        )
        sample_2 = TrainingSample(
            labeled_data=LabeledData(label_text="var", language_iso="en"), segment_selector_texts=["2/ var end"]
        )

        extraction_data = ExtractionData(samples=[sample_1, sample_2], extraction_identifier=extraction_identifier)

        regex_method = RegexSubtractionMethod(extraction_identifier)

        regex_method.train(extraction_data)

        texts = ["3/ one end", "4/ two end"]
        predictions = regex_method.predict([PredictionSample.from_text(text) for text in texts])
        self.assertEqual(2, len(predictions))
        self.assertEqual("one", predictions[0])
        self.assertEqual("two", predictions[1])

    def test_predict_void(self):
        sample_1 = TrainingSample(labeled_data=LabeledData(label_text="124", language_iso=""), segment_selector_texts=[""])
        sample_2 = TrainingSample(labeled_data=LabeledData(label_text="344", language_iso=""), segment_selector_texts=[])

        extraction_data = ExtractionData(samples=[sample_1, sample_2], extraction_identifier=extraction_identifier)

        regex_method = RegexSubtractionMethod(extraction_identifier)

        regex_method.train(extraction_data)

        predictions = regex_method.predict([PredictionSample.from_text("")])
        self.assertEqual(1, len(predictions))
        self.assertEqual("", predictions[0])
