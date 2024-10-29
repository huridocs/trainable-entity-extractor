from unittest import TestCase

from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.LabeledData import LabeledData
from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.data.TrainingSample import TrainingSample
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.SameInputOutputMethod import SameInputOutputMethod

extraction_identifier = ExtractionIdentifier(run_name="test", extraction_name="test")


class TestSameInputMethod(TestCase):
    def test_performance_100(self):
        sample = TrainingSample(
            labeled_data=LabeledData(label_text="a b c", language_iso="en"), segment_selector_texts=["a b c"]
        )

        extraction_data = ExtractionData(samples=[sample], extraction_identifier=extraction_identifier)

        same_input_output_method = SameInputOutputMethod(extraction_identifier)
        self.assertEqual(100, same_input_output_method.performance(extraction_data, extraction_data))

    def test_performance_100_with_multiline(self):
        label_text = """Albania, Algeria, Argentina, Bolivia (Plurinational State of), Brazil, Congo, Côte d’Ivoire,
                        El Salvador, Estonia, France, Gabon, Germany, Ireland, Kazakhstan, Latvia, Mexico,
                        Montenegro, Namibia, Netherlands, Paraguay, Portugal, Sierra Leone, South Africa, the
                        former Yugoslav Republic of Macedonia, United Kingdom of Great Britain and Northern
                        Ireland, Venezuela (Bolivarian Republic of)"""

        tags_text = [
            "Albania, Algeria, Argentina, Bolivia (Plurinational State of), Brazil, Congo, Côte d’Ivoire, ",
            "El  Salvador,  Estonia,  France,  Gabon,  Germany,  Ireland,  Kazakhstan,  Latvia,  Mexico, ",
            "Montenegro,  Namibia,  Netherlands,  Paraguay,  Portugal,  Sierra  Leone,  South  Africa,  ",
            "the former  Yugoslav  Republic  of  Macedonia,  United  Kingdom  of  Great  Britain  and",
            "Northern Ireland, Venezuela (Bolivarian Republic of)",
        ]

        sample = TrainingSample(
            labeled_data=LabeledData(label_text=label_text, language_iso="en"), segment_selector_texts=tags_text
        )

        extraction_data = ExtractionData(samples=[sample], extraction_identifier=extraction_identifier)

        same_input_output_method = SameInputOutputMethod(extraction_identifier)
        self.assertEqual(100, same_input_output_method.performance(extraction_data, extraction_data))

    def test_performance_50(self):
        sample_1 = TrainingSample(
            labeled_data=LabeledData(label_text="a b c", language_iso="en"), segment_selector_texts=["a b c"]
        )
        sample_2 = TrainingSample(
            labeled_data=LabeledData(label_text="2", language_iso="en"), segment_selector_texts=["a b c"]
        )

        extraction_data = ExtractionData(samples=[sample_1] + [sample_2], extraction_identifier=extraction_identifier)

        same_input_output_method = SameInputOutputMethod(extraction_identifier)

        self.assertEqual(50, same_input_output_method.performance(extraction_data, extraction_data))

    def test_predict(self):
        same_input_output_method = SameInputOutputMethod(extraction_identifier)
        predictions = same_input_output_method.predict([PredictionSample.from_text("test 1")])

        self.assertEqual(["test 1"], predictions)
