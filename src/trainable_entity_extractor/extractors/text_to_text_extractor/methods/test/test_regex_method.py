from unittest import TestCase

from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.LabeledData import LabeledData
from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.data.TrainingSample import TrainingSample
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.RegexMethod import RegexMethod

extraction_identifier = ExtractionIdentifier(run_name="test", extraction_name="test")


class TestRegexMethod(TestCase):
    def test_performance_100(self):
        sample = TrainingSample(labeled_data=LabeledData(label_text="12", language_iso="en"), tags_texts=["one 12"])

        extraction_data = ExtractionData(samples=[sample for _ in range(6)], extraction_identifier=extraction_identifier)
        regex_method = RegexMethod(extraction_identifier)

        self.assertEqual(100, regex_method.performance(extraction_data, extraction_data))

    def test_performance_0(self):
        sample = TrainingSample(labeled_data=LabeledData(label_text="12", language_iso="en"), tags_texts=["one two"])

        extraction_data = ExtractionData(samples=[sample for _ in range(6)], extraction_identifier=extraction_identifier)
        regex_method = RegexMethod(extraction_identifier)

        self.assertEqual(0, regex_method.performance(extraction_data, extraction_data))

    def test_performance_75(self):
        sample_1 = [TrainingSample(labeled_data=LabeledData(label_text="12", language_iso="en"), tags_texts=["one 12"])]
        sample_2 = [TrainingSample(labeled_data=LabeledData(label_text="no regex", language_iso="en"), tags_texts=["one"])]
        extraction_data = ExtractionData(samples=sample_1 * 3 + sample_2, extraction_identifier=extraction_identifier)

        regex_method = RegexMethod(extraction_identifier)

        self.assertEqual(75, regex_method.performance(extraction_data, extraction_data))

    def test_performance_no_samples(self):
        regex_method = RegexMethod(extraction_identifier)
        extraction_data = ExtractionData(samples=[], extraction_identifier=extraction_identifier)
        self.assertEqual(0, regex_method.performance(extraction_data, extraction_data))

    def test_performance_one_sample(self):
        sample = [TrainingSample(labeled_data=LabeledData(label_text="12", language_iso="en"), tags_texts=["one 12"])]
        extraction_data = ExtractionData(samples=sample, extraction_identifier=extraction_identifier)

        regex_method = RegexMethod(extraction_identifier)
        self.assertEqual(100, regex_method.performance(extraction_data, extraction_data))

    def test_predict(self):
        sample_1 = [TrainingSample(labeled_data=LabeledData(label_text="12", language_iso="en"), tags_texts=[""])]
        sample_2 = [TrainingSample(labeled_data=LabeledData(label_text="34", language_iso="en"), tags_texts=[""])]
        extraction_data = ExtractionData(samples=sample_1 + sample_2, extraction_identifier=extraction_identifier)

        regex_method = RegexMethod(extraction_identifier)

        regex_method.train(extraction_data)
        texts = ["one 12", "13", "14 foo"]
        predictions = regex_method.predict([PredictionSample.from_text(text) for text in texts])
        self.assertEqual(3, len(predictions))
        self.assertEqual("12", predictions[0])
        self.assertEqual("13", predictions[1])
        self.assertEqual("14", predictions[2])

    def test_predict_void(self):
        sample_1 = [TrainingSample(labeled_data=LabeledData(label_text="124", language_iso="en"), tags_texts=[""])]
        sample_2 = [TrainingSample(labeled_data=LabeledData(label_text="344", language_iso="en"), tags_texts=[""])]
        extraction_data = ExtractionData(samples=sample_1 + sample_2, extraction_identifier=extraction_identifier)

        regex_method = RegexMethod(extraction_identifier)

        regex_method.train(extraction_data)

        texts = ["14 foo", ""]
        predictions = regex_method.predict([PredictionSample.from_texts(texts)])
        self.assertEqual(1, len(predictions))
        self.assertEqual("", predictions[0])

    def test_regex_when_empty_labels(self):
        sample_1 = [TrainingSample(labeled_data=LabeledData(label_text="123", language_iso="en"), tags_texts=[""])]
        sample_2 = [TrainingSample(labeled_data=LabeledData(label_text="321", language_iso="en"), tags_texts=[""])]
        empty_labels = [TrainingSample(labeled_data=LabeledData(label_text="", language_iso="en"), tags_texts=[""])] * 100

        extraction_data = ExtractionData(
            samples=sample_1 + empty_labels + sample_2, extraction_identifier=extraction_identifier
        )

        regex_method = RegexMethod(extraction_identifier)

        regex_method.train(extraction_data)

        texts = ["foo 555 var"]
        predictions = regex_method.predict([PredictionSample.from_texts(texts)])
        self.assertEqual(1, len(predictions))
        self.assertEqual("555", predictions[0])

    def test_retrain(self):
        sample = [TrainingSample(labeled_data=LabeledData(label_text="1", language_iso="en"), tags_texts=[""])]
        extraction_data = ExtractionData(samples=sample, extraction_identifier=extraction_identifier)

        regex_method = RegexMethod(extraction_identifier)
        regex_method.train(extraction_data)

        sample = [TrainingSample(labeled_data=LabeledData(label_text="111", language_iso="en"), tags_texts=["one 12"])]
        extraction_data = ExtractionData(samples=sample, extraction_identifier=extraction_identifier)

        regex_method.train(extraction_data)

        predictions = regex_method.predict([PredictionSample.from_text("111")])
        self.assertEqual(1, len(predictions))
        self.assertEqual("111", predictions[0])
