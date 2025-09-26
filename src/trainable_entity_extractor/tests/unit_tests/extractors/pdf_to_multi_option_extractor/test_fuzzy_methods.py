from unittest import TestCase
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FastSegmentSelectorFuzzy95 import (
    FastSegmentSelectorFuzzy95,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyAll100 import (
    FuzzyAll100,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyAll75 import (
    FuzzyAll75,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyCommas import (
    FuzzyCommas,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyFirst import (
    FuzzyFirst,
)


class TestFuzzyMethods(TestCase):
    TENANT = "unit_test"
    extraction_id = "TestFuzzyMethods"

    def test_fuzzy_all_100(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="item 1"), Option(id="2", label="item 2"), Option(id="3", label="item 10")]

        pdf_data_1 = PdfData.from_texts(["blah. item 1. blah"])
        pdf_data_2 = PdfData.from_texts(["blah. item 10. blah"])
        pdf_data_3 = PdfData.from_texts(["blah. item 10, item 1. blah"])

        prediction_samples = [
            PredictionSample(pdf_data=pdf_data_1, entity_name="test"),
            PredictionSample(pdf_data=pdf_data_2, entity_name="test"),
            PredictionSample(pdf_data=pdf_data_3, entity_name="test"),
        ]

        prediction_samples_data = PredictionSamplesData(
            multi_value=True, options=options, prediction_samples=prediction_samples
        )

        predictions = FuzzyAll100().set_extraction_identifier(extraction_identifier).predict(prediction_samples_data)

        self.assertEqual(3, len(predictions))
        self.assertEqual([Value(id="1", label="item 1", segment_text="blah. item 1. blah")], predictions[0])
        self.assertEqual([Value(id="3", label="item 10", segment_text="blah. item 10. blah")], predictions[1])
        self.assertTrue(Value(id="1", label="item 1") in predictions[2])
        self.assertTrue(Value(id="2", label="item 2") not in predictions[2])
        self.assertTrue(Value(id="3", label="item 10") in predictions[2])

    def test_fuzzy_commas(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="item 1"), Option(id="2", label="item 2"), Option(id="10", label="item 10")]

        pdf_data_1 = PdfData.from_texts(["blah, item 1, 2 item, item 3, blah"])
        pdf_data_2 = PdfData.from_texts(["blah, 10 item, item 1, blah"])

        samples = [
            TrainingSample(pdf_data=pdf_data_1, labeled_data=LabeledData(values=[options[0]])),
            TrainingSample(pdf_data=pdf_data_2, labeled_data=LabeledData(values=[options[2]])),
        ]

        multi_option_data = ExtractionData(
            multi_value=True, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        prediction_samples = [
            PredictionSample(pdf_data=pdf_data_1, entity_name="test"),
            PredictionSample(pdf_data=pdf_data_2, entity_name="test"),
        ]

        prediction_samples_data = PredictionSamplesData(
            multi_value=True, options=options, prediction_samples=prediction_samples
        )

        method = FuzzyCommas().set_extraction_identifier(extraction_identifier)
        method.train(multi_option_data)
        predictions = method.predict(prediction_samples_data)

        self.assertEqual(2, len(predictions))

        self.assertTrue(Value(id="1", label="item 1") in predictions[0])
        self.assertTrue(Value(id="2", label="item 2") in predictions[0])
        self.assertTrue(Value(id="10", label="item 10") not in predictions[0])

        self.assertTrue(Value(id="1", label="item 1") in predictions[1])
        self.assertTrue(Value(id="2", label="item 2") not in predictions[1])
        self.assertTrue(Value(id="10", label="item 10") in predictions[1])

    def test_fuzzy_commas_aliases(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="  United Kingdom  ")]

        pdf_data_1 = PdfData.from_texts(
            ["blah,  United Kingdom of Great Britain and Northern Ireland  , 2 item, item 3, blah"]
        )

        pdf_data_1.pdf_data_segments[0].ml_label = 1

        samples = [
            TrainingSample(pdf_data=pdf_data_1, labeled_data=LabeledData(values=[options[0]])),
        ]

        multi_option_data = ExtractionData(
            multi_value=True, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        prediction_samples = [
            PredictionSample(pdf_data=pdf_data_1, entity_name="test"),
        ]

        prediction_samples_data = PredictionSamplesData(
            multi_value=True, options=options, prediction_samples=prediction_samples
        )

        method = FuzzyCommas().set_extraction_identifier(extraction_identifier)
        method.train(multi_option_data)
        predictions = method.predict(prediction_samples_data)

        self.assertEqual(1, len(predictions))

        self.assertTrue(Value(id="1", label="  United Kingdom  ") in predictions[0])

    def test_fast_segment_selector_fuzzy_95(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="item 1"), Option(id="2", label="item 2"), Option(id="10", label="item 10")]

        text = """No matter the scale or scope of partnership, we at HURIDOCS approach the task in a way that emphasises the following values: collaboration, purpose, safety, humanity and adaptability.
We are a human rights organisation too, and our ultimate vision is a world where all people’s dignity and freedom are protected. As such, if we see that our expertise or our tool Uwazi isn’t an ideal fit for your project, we’ll tell you so and do our best to refer you to allies who can help."""

        pdf_data_1 = PdfData.from_texts([text, "mark 1", "item 1, item 2, item 10", text])
        pdf_data_2 = PdfData.from_texts(["foo", "mark 1", "item 2", text])
        pdf_data_3 = PdfData.from_texts(["foo", "var", "mark 1", "item 10", text])

        samples = [
            TrainingSample(pdf_data=pdf_data_1, labeled_data=LabeledData(values=[options[0], options[1], options[2]])),
            TrainingSample(pdf_data=pdf_data_2, labeled_data=LabeledData(values=[options[1]])),
            TrainingSample(pdf_data=pdf_data_3, labeled_data=LabeledData(values=[options[2]])),
        ] * 5

        multi_option_data = ExtractionData(
            multi_value=True, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        prediction_samples = [
            PredictionSample(pdf_data=pdf_data_1, entity_name="test"),
            PredictionSample(pdf_data=pdf_data_2, entity_name="test"),
            PredictionSample(pdf_data=pdf_data_3, entity_name="test"),
        ] * 5

        prediction_samples_data = PredictionSamplesData(
            multi_value=True, options=options, prediction_samples=prediction_samples
        )

        fast_segment_selector_fuzzy = FastSegmentSelectorFuzzy95().set_extraction_identifier(extraction_identifier)
        fast_segment_selector_fuzzy.set_parameters(multi_option_data)
        fast_segment_selector_fuzzy.train(multi_option_data)
        predictions = fast_segment_selector_fuzzy.predict(prediction_samples_data)

        self.assertEqual(15, len(predictions))

        self.assertTrue(Value(id="1", label="item 1") in predictions[0])
        self.assertTrue(Value(id="2", label="item 2") in predictions[0])
        self.assertTrue(Value(id="10", label="item 10") in predictions[0])

        self.assertTrue(Value(id="1", label="item 1") not in predictions[1])
        self.assertTrue(Value(id="2", label="item 2") in predictions[1])
        self.assertTrue(Value(id="10", label="item 10") not in predictions[1])

        self.assertTrue(Value(id="1", label="item 1") not in predictions[2])
        self.assertTrue(Value(id="2", label="item 2") not in predictions[2])
        self.assertTrue(Value(id="10", label="item 10") in predictions[2])

    def test_fuzzy_all_75(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="item 1"), Option(id="2", label="item 2"), Option(id="3", label="item 10")]

        pdf_data_1 = PdfData.from_texts(["blah. item 1. blah"])

        prediction_samples = [
            PredictionSample(pdf_data=pdf_data_1, entity_name="test"),
        ]

        prediction_samples_data = PredictionSamplesData(
            multi_value=True, options=options, prediction_samples=prediction_samples
        )

        predictions = FuzzyAll75().set_extraction_identifier(extraction_identifier).predict(prediction_samples_data)

        self.assertEqual(1, len(predictions))
        self.assertTrue(Value(id="1", label="item 1") in predictions[0])
        self.assertTrue(Value(id="2", label="item 2") not in predictions[0])
        self.assertTrue(Value(id="3", label="item 10") in predictions[0])

    def test_fuzzy_first(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="item 1"), Option(id="2", label="item 2"), Option(id="3", label="item 10")]

        pdf_data_1 = PdfData.from_texts(["blah. item 1. blah"])
        pdf_data_2 = PdfData.from_texts(["blah. item 10. blah"])
        pdf_data_3 = PdfData.from_texts(["blah. item 10, item 1. blah"])

        prediction_samples = [
            PredictionSample(pdf_data=pdf_data_1, entity_name="test"),
            PredictionSample(pdf_data=pdf_data_2, entity_name="test"),
            PredictionSample(pdf_data=pdf_data_3, entity_name="test"),
        ]

        prediction_samples_data = PredictionSamplesData(
            multi_value=True, options=options, prediction_samples=prediction_samples
        )

        predictions = FuzzyFirst().set_extraction_identifier(extraction_identifier).predict(prediction_samples_data)

        self.assertEqual(3, len(predictions))
        self.assertEqual([Value(id="1", label="item 1", segment_text="blah. item 1. blah")], predictions[0])
        self.assertEqual([Value(id="3", label="item 10", segment_text="blah. item 10. blah")], predictions[1])
        self.assertTrue(Value(id="1", label="item 1") not in predictions[2])
        self.assertTrue(Value(id="2", label="item 2") not in predictions[2])
        self.assertTrue(Value(id="3", label="item 10") in predictions[2])
