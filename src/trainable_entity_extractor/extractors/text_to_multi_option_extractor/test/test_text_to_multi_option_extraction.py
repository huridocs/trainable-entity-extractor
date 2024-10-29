from unittest import TestCase
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.LabeledData import LabeledData
from trainable_entity_extractor.data.Option import Option
from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.data.TrainingSample import TrainingSample
from trainable_entity_extractor.extractors.text_to_multi_option_extractor.TextToMultiOptionExtractor import (
    TextToMultiOptionExtractor,
)


class TestTextToMultiOptionExtraction(TestCase):
    TENANT = "multi_option_extraction_test"
    extraction_id = "extraction_id"

    def test_is_valid(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name="other")
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        samples_text = [TrainingSample(labeled_data=LabeledData(source_text="1"))]
        samples_no_text = [TrainingSample(labeled_data=LabeledData(source_text=""))]

        multi_option_extraction = TextToMultiOptionExtractor(extraction_identifier)
        no_options = ExtractionData(extraction_identifier=extraction_identifier, samples=samples_text)
        no_text = ExtractionData(extraction_identifier=extraction_identifier, options=options, samples=samples_no_text)
        valid_extraction_data = ExtractionData(
            extraction_identifier=extraction_identifier, options=options, samples=samples_text
        )

        self.assertFalse(multi_option_extraction.can_be_used(no_options))
        self.assertFalse(multi_option_extraction.can_be_used(no_text))
        self.assertTrue(multi_option_extraction.can_be_used(valid_extraction_data))

    def test_single_value(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        samples = [
            TrainingSample(segment_selector_texts=["point 1"], labeled_data=LabeledData(values=[options[0]])),
            TrainingSample(segment_selector_texts=["point 2"], labeled_data=LabeledData(values=[options[1]])),
            TrainingSample(segment_selector_texts=["point 3  point 2"], labeled_data=LabeledData(values=[options[2]])),
        ]

        multi_option_data = ExtractionData(
            multi_value=False, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        multi_option_extraction = TextToMultiOptionExtractor(extraction_identifier)
        multi_option_extraction.create_model(multi_option_data)

        prediction_sample_1 = PredictionSample(segment_selector_texts=["point 1"], entity_name="entity_name_1")
        prediction_sample_3 = PredictionSample(segment_selector_texts=["point 3 point 2"], entity_name="entity_name_3")
        suggestions = multi_option_extraction.get_suggestions([prediction_sample_1, prediction_sample_3])

        self.assertEqual(2, len(suggestions))
        self.assertEqual([Option(id="1", label="1")], suggestions[0].values)
        self.assertEqual("entity_name_1", suggestions[0].entity_name)
        self.assertEqual([Option(id="3", label="3")], suggestions[1].values)
        self.assertEqual("entity_name_3", suggestions[1].entity_name)

    def test_multi_value(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        samples = [
            TrainingSample(
                segment_selector_texts=["point 1 point 2"], labeled_data=LabeledData(values=[options[0], options[1]])
            ),
            TrainingSample(segment_selector_texts=["point 2"], labeled_data=LabeledData(values=[options[1]])),
            TrainingSample(
                segment_selector_texts=["point 3 point 1"], labeled_data=LabeledData(values=[options[2], options[0]])
            ),
        ]

        multi_option_data = ExtractionData(
            multi_value=True, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        multi_option_extraction = TextToMultiOptionExtractor(extraction_identifier)
        multi_option_extraction.create_model(multi_option_data)

        prediction_sample_1 = PredictionSample(segment_selector_texts=["point 1 point 2"], entity_name="entity_name_1")
        prediction_sample_3 = PredictionSample(segment_selector_texts=["point 3 point 1"], entity_name="entity_name_3")
        suggestions = multi_option_extraction.get_suggestions([prediction_sample_1, prediction_sample_3])

        self.assertEqual(2, len(suggestions))
        self.assertTrue(Option(id="1", label="1") in suggestions[0].values)
        self.assertTrue(Option(id="2", label="2") in suggestions[0].values)
        self.assertTrue(Option(id="3", label="3") not in suggestions[0].values)
        self.assertEqual("entity_name_1", suggestions[0].entity_name)

        self.assertTrue(Option(id="3", label="3") in suggestions[1].values)
        self.assertTrue(Option(id="2", label="2") not in suggestions[1].values)
        self.assertTrue(Option(id="1", label="1") in suggestions[1].values)
        self.assertEqual("entity_name_3", suggestions[1].entity_name)

    def test_tf_idf(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        samples = [
            TrainingSample(
                segment_selector_texts=["point one point two"], labeled_data=LabeledData(values=[options[0], options[1]])
            ),
            TrainingSample(segment_selector_texts=["point two"], labeled_data=LabeledData(values=[options[1]])),
            TrainingSample(
                segment_selector_texts=["point three point one"], labeled_data=LabeledData(values=[options[2], options[0]])
            ),
        ]

        multi_option_data = ExtractionData(
            multi_value=True, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        multi_option_extraction = TextToMultiOptionExtractor(extraction_identifier)
        multi_option_extraction.create_model(multi_option_data)

        prediction_sample_1 = PredictionSample(segment_selector_texts=["point one point two"], entity_name="entity_name_1")
        prediction_sample_3 = PredictionSample(segment_selector_texts=["point three point one"], entity_name="entity_name_3")
        suggestions = multi_option_extraction.get_suggestions([prediction_sample_1, prediction_sample_3])

        self.assertEqual(2, len(suggestions))
        self.assertTrue(Option(id="1", label="1") in suggestions[0].values)
        self.assertTrue(Option(id="2", label="2") in suggestions[0].values)
        self.assertTrue(Option(id="3", label="3") not in suggestions[0].values)
        self.assertTrue(Option(id="3", label="3") in suggestions[1].values)
        self.assertTrue(Option(id="2", label="2") not in suggestions[1].values)
        self.assertTrue(Option(id="1", label="1") in suggestions[1].values)

    def test_no_prediction_data(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        samples = [
            TrainingSample(segment_selector_texts=["1 2"], labeled_data=LabeledData(values=[options[0], options[1]])),
        ]

        multi_option_data = ExtractionData(
            multi_value=True, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        multi_option_extraction = TextToMultiOptionExtractor(extraction_identifier)
        multi_option_extraction.create_model(multi_option_data)

        suggestions = multi_option_extraction.get_suggestions([])

        self.assertEqual(0, len(suggestions))
