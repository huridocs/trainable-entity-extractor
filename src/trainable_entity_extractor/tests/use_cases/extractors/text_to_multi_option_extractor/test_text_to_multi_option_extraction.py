from unittest import TestCase
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.TextToMultiOptionExtractor import (
    TextToMultiOptionExtractor,
)


class TestTextToMultiOptionExtraction(TestCase):
    TENANT = "default"
    extraction_id = "multi_option_extraction_test"

    def test_can_be_used(self):
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
        self.assertTrue(multi_option_extraction.can_be_used(no_text))
        self.assertTrue(multi_option_extraction.can_be_used(valid_extraction_data))

    def test_single_value(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        samples = [
            TrainingSample(labeled_data=LabeledData(values=[options[0]], source_text="point 1")),
            TrainingSample(labeled_data=LabeledData(values=[options[1]], source_text="point 2")),
            TrainingSample(labeled_data=LabeledData(values=[options[2]], source_text="point 3 point 2")),
        ]

        multi_option_data = ExtractionData(
            multi_value=False, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        multi_option_extraction = TextToMultiOptionExtractor(extraction_identifier)
        multi_option_extraction.create_model(multi_option_data)

        prediction_sample_1 = PredictionSample(source_text="point 1", entity_name="entity_name_1")
        prediction_sample_3 = PredictionSample(source_text="point 3 point 2", entity_name="entity_name_3")
        suggestions = multi_option_extraction.get_suggestions([prediction_sample_1, prediction_sample_3])

        self.assertEqual(2, len(suggestions))
        self.assertEqual([Option(id="1", label="1")], suggestions[0].values)
        self.assertEqual("entity_name_1", suggestions[0].entity_name)
        self.assertEqual([Option(id="3", label="3")], suggestions[1].values)
        self.assertEqual("entity_name_3", suggestions[1].entity_name)

    def test_first_word_regex(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name="first_word_regex")
        options = [Option(id="yes", label="yes"), Option(id="no", label="no")]

        samples = [
            TrainingSample(
                labeled_data=LabeledData(
                    values=[options[0]],
                    source_text="""139.1
Finalize the ratification of the ILO Violence and Harassment
Convention, 2019 (No. 190) (Democratic Republic of the Congo);""",
                )
            ),
            TrainingSample(
                labeled_data=LabeledData(
                    values=[options[0]],
                    source_text="""139.6
Continue enhancing its national mechanism for the implementation,
reporting and follow-up of human rights recommendations (Angola);""",
                )
            ),
            TrainingSample(
                labeled_data=LabeledData(
                    values=[options[0]],
                    source_text="""140.30
Take specific measures, including strengthening the legal framework,
to eliminate discrimination, hate speech and violence against lesbian, bisexual
and transgender women, including by prosecuting and adequately punishing
perpetrators, and adopt awareness-raising measures to address stigma within
society (Liechtenstein);""",
                )
            ),
            TrainingSample(
                labeled_data=LabeledData(
                    values=[options[0]],
                    source_text="""
            140.46
Consider introducing a universal basic income in order to better
combat poverty and reduce inequalities, and improve the existing social
protection system (Haiti);""",
                )
            ),
            TrainingSample(
                labeled_data=LabeledData(
                    values=[options[1]],
                    source_text="""62.
Viet Nam commended the strong commitment of Fiji to the advancement of the
rights of women and children, especially in the context of mitigating the negative impact of
climate change.""",
                )
            ),
            TrainingSample(
                labeled_data=LabeledData(
                    values=[options[1]],
                    source_text="""35.
The Republic of Korea appreciated efforts to protect persons with disabilities and
welcomed the ratification of the remaining six human rights treaties.""",
                )
            ),
            TrainingSample(
                labeled_data=LabeledData(
                    values=[options[1]],
                    source_text="""
            Report of the Working Group on the Universal Periodic
Review*""",
                )
            ),
            TrainingSample(
                labeled_data=LabeledData(
                    values=[options[1]],
                    source_text="""
            Original: English""",
                )
            ),
            TrainingSample(
                labeled_data=LabeledData(
                    values=[options[1]],
                    source_text="""
            Human Rights Council
Forty-third session
24 February–20 March 2020
Agenda item 6
Universal periodic review""",
                )
            ),
        ]

        multi_option_data = ExtractionData(
            multi_value=False, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        multi_option_extraction = TextToMultiOptionExtractor(extraction_identifier)
        multi_option_extraction.create_model(multi_option_data)

        prediction_sample_1 = PredictionSample(
            source_text="""139.16
Continue working with all stakeholders, including the International
Labour Organization, to progress issues raised in the joint implementation
report (Australia);""",
            entity_name="entity_name_1",
        )
        prediction_sample_2 = PredictionSample(
            source_text="""H.E. Ms. Nazhat Shameem Khan, Ambassador and Permanent Representative;""",
            entity_name="entity_name_2",
        )
        suggestions = multi_option_extraction.get_suggestions([prediction_sample_1, prediction_sample_2])

        self.assertEqual(2, len(suggestions))
        self.assertEqual([options[0]], suggestions[0].values)
        self.assertEqual([options[1]], suggestions[1].values)

    def test_multi_value(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="0", label="0"), Option(id="1", label="1"), Option(id="2", label="2")]

        samples = [
            TrainingSample(labeled_data=LabeledData(values=[options[0], options[1]], source_text="point 0 point 1")),
            TrainingSample(labeled_data=LabeledData(values=[options[1]], source_text="point 1")),
            TrainingSample(labeled_data=LabeledData(values=[options[2], options[0]], source_text="point 2 point 0")),
            TrainingSample(labeled_data=LabeledData(values=[options[2]], source_text="point 2")),
            TrainingSample(labeled_data=LabeledData(values=[options[1], options[2]], source_text="point 1 point 2")),
        ]

        multi_option_data = ExtractionData(
            multi_value=True, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        multi_option_extraction = TextToMultiOptionExtractor(extraction_identifier)
        multi_option_extraction.create_model(multi_option_data)

        prediction_sample_1 = PredictionSample(source_text="point 0 point 1", entity_name="entity_name_1")
        prediction_sample_3 = PredictionSample(source_text="point 2 point 0", entity_name="entity_name_3")
        suggestions = multi_option_extraction.get_suggestions([prediction_sample_1, prediction_sample_3])

        self.assertEqual(2, len(suggestions))
        self.assertTrue(Option(id="0", label="0") in suggestions[0].values)
        self.assertTrue(Option(id="1", label="1") in suggestions[0].values)
        self.assertTrue(Option(id="2", label="2") not in suggestions[0].values)
        self.assertEqual("entity_name_1", suggestions[0].entity_name)

        self.assertTrue(Option(id="0", label="0") in suggestions[1].values)
        self.assertTrue(Option(id="1", label="1") not in suggestions[1].values)
        self.assertTrue(Option(id="2", label="2") in suggestions[1].values)
        self.assertEqual("entity_name_3", suggestions[1].entity_name)

    def test_no_prediction_data(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        samples = [TrainingSample(labeled_data=LabeledData(values=[options[0], options[1]], source_text="1 2"))]

        multi_option_data = ExtractionData(
            multi_value=True, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        multi_option_extraction = TextToMultiOptionExtractor(extraction_identifier)
        multi_option_extraction.create_model(multi_option_data)

        suggestions = multi_option_extraction.get_suggestions([])

        self.assertEqual(0, len(suggestions))
