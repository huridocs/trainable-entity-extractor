from unittest import TestCase

from trainable_entity_extractor.adapters.ExtractorLogger import ExtractorLogger
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.TextToMultiOptionExtractor import (
    TextToMultiOptionExtractor,
)


class TestTextToMultiOptionExtraction(TestCase):
    TENANT = "unit_test"
    extraction_id = "multi_option_extraction_test"

    def setUp(self):
        self.logger = ExtractorLogger()

    def test_can_be_used(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name="other")
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        samples_text = [TrainingSample(labeled_data=LabeledData(source_text="1"))]
        samples_no_text = [TrainingSample(labeled_data=LabeledData(source_text=""))]

        multi_option_extraction = TextToMultiOptionExtractor(extraction_identifier, self.logger)
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

        multi_option_extraction = TextToMultiOptionExtractor(extraction_identifier, self.logger)

        method = "TextFuzzyFirst"
        job = TrainableEntityExtractorJob(
            run_name=self.TENANT,
            extraction_name=self.extraction_id,
            extractor_name="TextToMultiOptionExtractor",
            method_name=method,
            gpu_needed=False,
            timeout=60,
        )
        success, error_msg = multi_option_extraction.train_one_method(job, multi_option_data)
        self.assertTrue(success, f"Training failed: {error_msg}")

        prediction_sample_1 = PredictionSample(source_text="point 1", entity_name="entity_name_1")
        prediction_sample_3 = PredictionSample(source_text="point 3 point 2", entity_name="entity_name_3")
        prediction_samples_data = PredictionSamplesData(
            prediction_samples=[prediction_sample_1, prediction_sample_3], options=options, multi_value=False
        )
        suggestions = multi_option_extraction.get_suggestions(method, prediction_samples_data)

        self.assertEqual(2, len(suggestions))
        self.assertEqual([Value(id="1", label="1", segment_text="point 1")], suggestions[0].values)
        self.assertEqual("entity_name_1", suggestions[0].entity_name)
        self.assertEqual([Value(id="3", label="3", segment_text="point 3 point 2")], suggestions[1].values)
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

        multi_option_extraction = TextToMultiOptionExtractor(extraction_identifier, self.logger)

        method = "FirstWordRegex"
        job = TrainableEntityExtractorJob(
            run_name=self.TENANT,
            extraction_name="first_word_regex",
            extractor_name="TextToMultiOptionExtractor",
            method_name=method,
            gpu_needed=False,
            timeout=60,
        )
        success, error_msg = multi_option_extraction.train_one_method(job, multi_option_data)
        self.assertTrue(success, f"Training failed: {error_msg}")

        source_text_1 = """139.16
Continue working with all stakeholders, including the International
Labour Organization, to progress issues raised in the joint implementation
report (Australia);"""
        prediction_sample_1 = PredictionSample(
            source_text=source_text_1,
            entity_name="entity_name_1",
        )
        source_text_2 = """H.E. Ms. Nazhat Shameem Khan, Ambassador and Permanent Representative;"""
        prediction_sample_2 = PredictionSample(
            source_text=source_text_2,
            entity_name="entity_name_2",
        )
        prediction_samples_data = PredictionSamplesData(
            prediction_samples=[prediction_sample_1, prediction_sample_2], options=options, multi_value=False
        )
        suggestions = multi_option_extraction.get_suggestions(method, prediction_samples_data)

        self.assertEqual(2, len(suggestions))
        self.assertEqual(
            [Value(id=options[0].id, label=options[0].label, segment_text=source_text_1)], suggestions[0].values
        )
        self.assertEqual(
            [Value(id=options[1].id, label=options[1].label, segment_text=source_text_2)], suggestions[1].values
        )

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

        multi_option_extraction = TextToMultiOptionExtractor(extraction_identifier, self.logger)

        method = "TextFuzzyAll100"
        job = TrainableEntityExtractorJob(
            run_name=self.TENANT,
            extraction_name=self.extraction_id,
            extractor_name="TextToMultiOptionExtractor",
            method_name=method,
            gpu_needed=False,
            timeout=60,
        )
        success, error_msg = multi_option_extraction.train_one_method(job, multi_option_data)
        self.assertTrue(success, f"Training failed: {error_msg}")

        prediction_sample_1 = PredictionSample(source_text="point 0 point 1", entity_name="entity_name_1")
        prediction_sample_3 = PredictionSample(source_text="point 2 point 0", entity_name="entity_name_3")
        prediction_samples_data = PredictionSamplesData(
            prediction_samples=[prediction_sample_1, prediction_sample_3], options=options, multi_value=True
        )
        suggestions = multi_option_extraction.get_suggestions(method, prediction_samples_data)

        self.assertEqual(2, len(suggestions))
        self.assertTrue(Value(id="0", label="0", segment_text="point 0 point 1") in suggestions[0].values)
        self.assertTrue(Value(id="1", label="1", segment_text="point 0 point 1") in suggestions[0].values)
        self.assertTrue(Value(id="2", label="2", segment_text="point 0 point 1") not in suggestions[0].values)
        self.assertEqual("entity_name_1", suggestions[0].entity_name)

        self.assertTrue(Value(id="0", label="0", segment_text="point 2 point 0") in suggestions[1].values)
        self.assertTrue(Value(id="1", label="1", segment_text="point 2 point 0") not in suggestions[1].values)
        self.assertTrue(Value(id="2", label="2", segment_text="point 2 point 0") in suggestions[1].values)
        self.assertEqual("entity_name_3", suggestions[1].entity_name)

    def test_no_prediction_data(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        samples = [TrainingSample(labeled_data=LabeledData(values=[options[0], options[1]], source_text="1 2"))]

        multi_option_data = ExtractionData(
            multi_value=True, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        multi_option_extraction = TextToMultiOptionExtractor(extraction_identifier, self.logger)

        method = "NaiveTextToMultiOptionMethod"
        job = TrainableEntityExtractorJob(
            run_name=self.TENANT,
            extraction_name=self.extraction_id,
            extractor_name="TextToMultiOptionExtractor",
            method_name=method,
            gpu_needed=False,
            timeout=60,
        )
        success, error_msg = multi_option_extraction.train_one_method(job, multi_option_data)
        self.assertTrue(success, f"Training failed: {error_msg}")

        prediction_samples_data = PredictionSamplesData(prediction_samples=[], options=options, multi_value=True)
        suggestions = multi_option_extraction.get_suggestions(method, prediction_samples_data)

        self.assertEqual(0, len(suggestions))

    def test_get_train_test_sets_using_labels(self):
        # Test Case 1: Condition `len(all_samples) - len(test_set) < 8` is TRUE
        # This means the initial test set (from smallest groups) leaves too few for training.
        # Fallback: test_set becomes int(total_samples * 0.30) of all samples.
        self.subTest("Condition: train set too small (fallback to 30% test set)")
        s_c1_1 = TrainingSample(labeled_data=LabeledData(source_text="s_c1_1"))
        s_c1_2 = TrainingSample(labeled_data=LabeledData(source_text="s_c1_2"))
        s_c1_3 = TrainingSample(labeled_data=LabeledData(source_text="s_c1_3"))
        s_c1_4 = TrainingSample(labeled_data=LabeledData(source_text="s_c1_4"))
        s_c1_5 = TrainingSample(labeled_data=LabeledData(source_text="s_c1_5"))
        s_c1_6 = TrainingSample(labeled_data=LabeledData(source_text="s_c1_6"))
        s_c1_7 = TrainingSample(labeled_data=LabeledData(source_text="s_c1_7"))

        samples_by_labels_c1 = {
            "A": [s_c1_1, s_c1_2],  # Smallest group (or tied), 2 samples
            "B": [s_c1_3, s_c1_4, s_c1_5],  # 3 samples
            "C": [s_c1_6, s_c1_7],  # 2 samples
        }
        all_samples_c1_set = {s_c1_1, s_c1_2, s_c1_3, s_c1_4, s_c1_5, s_c1_6, s_c1_7}  # 7 unique samples

        # Expected logic walk-through for c1:
        # sorted_labels_by_samples_count: e.g., ["A", "C", "B"] or ["C", "A", "B"]
        # Loop 1 (label "A"): initial_test_set = {s_c1_1, s_c1_2}. size = 2.
        # 2 / 7 (total) approx 0.28. This is >= 0.10. Loop breaks.
        # Current test_set = {s_c1_1, s_c1_2}.
        # len(all_samples_c1_set) - len(test_set) = 7 - 2 = 5.
        # Condition `5 < 8` is TRUE.
        # test_set is reassigned: set(list(all_samples_c1_set)[:int(7 * 0.30)])
        # int(7 * 0.30) = int(2.1) = 2.
        # So, test_set will contain 2 samples, taken from the start of list(all_samples_c1_set).
        expected_test_size_c1 = int(len(all_samples_c1_set) * 0.30)  # Should be 2
        expected_train_size_c1 = len(all_samples_c1_set) - expected_test_size_c1  # Should be 5

        train_list_c1, test_list_c1 = TextToMultiOptionExtractor.get_train_test_sets_using_labels(samples_by_labels_c1)
        train_set_c1 = set(train_list_c1)
        test_set_c1 = set(test_list_c1)

        self.assertEqual(len(test_set_c1), expected_test_size_c1)
        self.assertEqual(len(train_set_c1), expected_train_size_c1)
        self.assertTrue(train_set_c1.isdisjoint(test_set_c1), "Train and test sets should be disjoint")
        self.assertEqual(
            train_set_c1 | test_set_c1, all_samples_c1_set, "Union of train and test sets should be all samples"
        )

        # Test Case 2: Condition `len(all_samples) - len(test_set) < 8` is FALSE
        # Sufficient samples for training after initial test_set creation from smallest groups.
        # Fallback: test_set = initial_test_set + update(list(all_samples)[:10%])
        self.subTest("Condition: train set large enough (initial test set + 10% update)")

        # Create 20 unique samples
        samples_list_c2 = [TrainingSample(labeled_data=LabeledData(source_text=f"s_c2_{i}")) for i in range(20)]
        all_samples_c2_set = set(samples_list_c2)  # 20 unique samples

        samples_by_labels_c2 = {
            "A": [samples_list_c2[0], samples_list_c2[1]],  # Smallest group, 2 samples (s_c2_0, s_c2_1)
            "B": [samples_list_c2[2], samples_list_c2[3], samples_list_c2[4]],  # 3 samples
            "C": samples_list_c2[5:],  # Remaining 15 samples
        }

        # Expected logic walk-through for c2:
        # sorted_labels_by_samples_count: ["A", "B", "C"]
        # Loop 1 (label "A"): initial_test_set_from_loop = {s_c2_0, s_c2_1}. size = 2.
        # 2 / 20 (total) = 0.1. This is >= 0.10. Loop breaks.
        # Current test_set from loop = {s_c2_0, s_c2_1}.
        # len(all_samples_c2_set) - len(test_set_from_loop) = 20 - 2 = 18.
        # Condition `18 < 8` is FALSE. Else branch is taken.
        # test_size_for_update = int(20 * 0.10) = 2.
        # test_set.update(list(all_samples_c2_set)[:2])
        # The final test_set will be {s_c2_0, s_c2_1} union {first 2 samples from list(all_samples_c2_set)}.
        # Size of test_set will be between 2 (if first 2 are s_c2_0, s_c2_1) and 4 (if first 2 are different).

        initial_smallest_group_samples_c2 = {samples_list_c2[0], samples_list_c2[1]}
        min_expected_test_size_c2 = len(initial_smallest_group_samples_c2)  # 2
        # Max size is if the 10% update adds completely new samples
        max_expected_test_size_c2 = len(initial_smallest_group_samples_c2) + int(len(all_samples_c2_set) * 0.10)  # 2 + 2 = 4

        train_list_c2, test_list_c2 = TextToMultiOptionExtractor.get_train_test_sets_using_labels(samples_by_labels_c2)
        train_set_c2 = set(train_list_c2)
        test_set_c2 = set(test_list_c2)

        self.assertTrue(train_set_c2.isdisjoint(test_set_c2), "Train and test sets should be disjoint")
        self.assertEqual(
            train_set_c2 | test_set_c2, all_samples_c2_set, "Union of train and test sets should be all samples"
        )

        self.assertTrue(
            initial_smallest_group_samples_c2.issubset(test_set_c2),
            "Test set should contain the initial smallest group samples",
        )
        self.assertGreaterEqual(
            len(test_set_c2), min_expected_test_size_c2, "Test set size should be at least the size of the smallest group"
        )
        self.assertLessEqual(
            len(test_set_c2), max_expected_test_size_c2, "Test set size should not exceed smallest group + 10% of total"
        )
