import os
import shutil
from os.path import join
from pathlib import Path

from unittest import TestCase

from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.LabeledData import LabeledData
from trainable_entity_extractor.data.PredictionData import PredictionData
from trainable_entity_extractor.data.SegmentBox import SegmentBox
from trainable_entity_extractor.FilterValidSegmentsPages import FilterValidSegmentsPages

DATA_FOLDER_PATH = (
    f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))}/models_data"
)


class TestFilterValidSegmentPages(TestCase):
    @staticmethod
    def get_labeled_data(labels_page_numbers: list[int], number_of_pages: int = None):
        if not number_of_pages:
            number_of_pages = max(labels_page_numbers)

        return LabeledData(
            tenant="",
            id="",
            xml_file_name="",
            language_iso="",
            label_text="",
            page_width=0,
            page_height=0,
            xml_segments_boxes=[
                SegmentBox(left=0, top=0, width=0, page_width=0, page_height=0, height=0, page_number=i + 1)
                for i in range(number_of_pages)
            ],
            label_segments_boxes=[
                SegmentBox(left=0, top=0, page_width=0, page_height=0, width=0, height=0, page_number=label_number)
                for label_number in labels_page_numbers
            ],
        )

    @staticmethod
    def get_prediction_data(page_numbers: list[int]):
        return PredictionData(
            tenant="",
            id="",
            xml_file_name="",
            page_width=0,
            page_height=0,
            xml_segments_boxes=[
                SegmentBox(left=0, top=0, width=0, page_width=0, page_height=0, height=0, page_number=page_number)
                for page_number in page_numbers
            ],
        )

    def test_get_valid_pages_one_document(self):
        tenant = "tenant_filter"
        extraction_id = "property_filter"

        shutil.rmtree(join(DATA_FOLDER_PATH, tenant), ignore_errors=True)

        filter_valid_segment_pages = FilterValidSegmentsPages(
            ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id)
        )
        labeled_data = self.get_labeled_data(labels_page_numbers=[1, 3, 6])

        pages_per_document = filter_valid_segment_pages.for_training([labeled_data])
        self.assertEqual([1, 2, 3, 4, 5, 6], pages_per_document[0])

    def test_get_valid_pages_no_label(self):
        tenant = "tenant_filter"
        extraction_id = "property_filter"

        shutil.rmtree(join(DATA_FOLDER_PATH, tenant), ignore_errors=True)

        filter_valid_segment_pages = FilterValidSegmentsPages(
            ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id)
        )
        labeled_data_1 = self.get_labeled_data(labels_page_numbers=[1], number_of_pages=10)
        labeled_data_2 = self.get_labeled_data(labels_page_numbers=[], number_of_pages=10)

        pages_per_document = filter_valid_segment_pages.for_training([labeled_data_1, labeled_data_2])
        self.assertEqual([1, 2], pages_per_document[1])

    def test_get_valid_pages_two_documents(self):
        tenant = "tenant_filter"
        extraction_id = "property_filter"

        shutil.rmtree(join(DATA_FOLDER_PATH, tenant), ignore_errors=True)

        filter_valid_segment_pages = FilterValidSegmentsPages(
            ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id)
        )
        labeled_data_1 = self.get_labeled_data(labels_page_numbers=[1, 3, 6])
        labeled_data_2 = self.get_labeled_data(labels_page_numbers=[2, 4])

        pages_per_document = filter_valid_segment_pages.for_training([labeled_data_1, labeled_data_2])
        self.assertEqual([1, 2, 3, 4, 5, 6], pages_per_document[0])
        self.assertEqual([1, 2, 3, 4], pages_per_document[1])

    def test_get_valid_pages_only_first_page(self):
        tenant = "tenant_filter"
        extraction_id = "property_filter"

        shutil.rmtree(join(DATA_FOLDER_PATH, tenant), ignore_errors=True)

        filter_valid_segment_pages = FilterValidSegmentsPages(
            ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id)
        )
        labeled_data_1 = self.get_labeled_data(labels_page_numbers=[1])
        labeled_data_2 = self.get_labeled_data(labels_page_numbers=[1])

        pages_per_document = filter_valid_segment_pages.for_training([labeled_data_1, labeled_data_2])
        self.assertEqual([1], pages_per_document[0])
        self.assertEqual([1], pages_per_document[1])

    def test_get_valid_pages_two_documents_more_variability(self):
        tenant = "tenant_filter"
        extraction_id = "property_filter"

        shutil.rmtree(join(DATA_FOLDER_PATH, tenant), ignore_errors=True)

        filter_valid_segment_pages = FilterValidSegmentsPages(
            ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id)
        )
        labeled_data_1 = self.get_labeled_data(labels_page_numbers=[2], number_of_pages=18)
        labeled_data_2 = self.get_labeled_data(labels_page_numbers=[6], number_of_pages=18)

        pages_per_document = filter_valid_segment_pages.for_training([labeled_data_1, labeled_data_2])
        self.assertEqual([1, 2, 3, 4, 5, 6, 7], pages_per_document[0])
        self.assertEqual([1, 2, 3, 4, 5, 6, 7], pages_per_document[1])

    def test_get_valid_pages_from_the_end(self):
        tenant = "tenant_filter"
        extraction_id = "property_filter"

        shutil.rmtree(join(DATA_FOLDER_PATH, tenant), ignore_errors=True)

        filter_valid_segment_pages = FilterValidSegmentsPages(
            ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id)
        )
        labeled_data_1 = self.get_labeled_data(labels_page_numbers=[4, 5])
        labeled_data_2 = self.get_labeled_data(labels_page_numbers=[10, 11, 12])

        pages_per_document = filter_valid_segment_pages.for_training([labeled_data_1, labeled_data_2])
        self.assertEqual([2, 3, 4, 5], pages_per_document[0])
        self.assertEqual([9, 10, 11, 12], pages_per_document[1])

    def test_get_valid_pages_from_the_end_different_page_numbers(self):
        tenant = "tenant_filter"
        extraction_id = "property_filter"

        shutil.rmtree(join(DATA_FOLDER_PATH, tenant), ignore_errors=True)

        filter_valid_segment_pages = FilterValidSegmentsPages(
            ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id)
        )
        labeled_data_1 = self.get_labeled_data(labels_page_numbers=[10, 11, 12, 13])
        labeled_data_2 = self.get_labeled_data(labels_page_numbers=[2])

        pages_per_document = filter_valid_segment_pages.for_training([labeled_data_1, labeled_data_2])
        self.assertEqual([9, 10, 11, 12, 13], pages_per_document[0])
        self.assertEqual([1, 2], pages_per_document[1])

    def test_get_valid_pages_two_documents_from_the_end_more_variability(self):
        tenant = "tenant_filter"
        extraction_id = "property_filter"

        shutil.rmtree(join(DATA_FOLDER_PATH, tenant), ignore_errors=True)

        filter_valid_segment_pages = FilterValidSegmentsPages(
            ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id)
        )
        labeled_data_1 = self.get_labeled_data(labels_page_numbers=[3], number_of_pages=4)
        labeled_data_2 = self.get_labeled_data(labels_page_numbers=[6], number_of_pages=6)

        pages_per_document = filter_valid_segment_pages.for_training([labeled_data_1, labeled_data_2])
        self.assertEqual([2, 3, 4], pages_per_document[0])
        self.assertEqual([4, 5, 6], pages_per_document[1])

    def test_get_valid_pages_for_prediction(self):
        tenant = "tenant_filter"
        extraction_id = "property_filter"

        shutil.rmtree(join(DATA_FOLDER_PATH, tenant), ignore_errors=True)

        filter_valid_segment_pages = FilterValidSegmentsPages(
            ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id)
        )
        labeled_data = self.get_labeled_data(labels_page_numbers=[1, 2, 3])
        prediction_data = self.get_prediction_data(page_numbers=[2])

        filter_valid_segment_pages.for_training([labeled_data])
        pages_per_document = filter_valid_segment_pages.for_prediction([prediction_data])
        self.assertEqual(1, len(pages_per_document))
        self.assertEqual([1, 2], pages_per_document[0])

    def test_get_valid_pages_for_prediction_from_the_end(self):
        tenant = "tenant_filter"
        extraction_id = "property_filter"

        shutil.rmtree(join(DATA_FOLDER_PATH, tenant), ignore_errors=True)

        filter_valid_segment_pages = FilterValidSegmentsPages(
            ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id)
        )
        labeled_data = self.get_labeled_data(labels_page_numbers=[10, 11, 12, 13])
        prediction_data = self.get_prediction_data(page_numbers=[2])

        filter_valid_segment_pages.for_training([labeled_data])
        pages_per_document = filter_valid_segment_pages.for_prediction([prediction_data])
        self.assertEqual(1, len(pages_per_document))
        self.assertEqual([1, 2], pages_per_document[0])

    def test_get_valid_pages_for_prediction_void(self):
        tenant = "tenant_filter"
        extraction_id = "property_filter"

        shutil.rmtree(join(DATA_FOLDER_PATH, tenant), ignore_errors=True)

        filter_valid_segment_pages = FilterValidSegmentsPages(
            ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id)
        )
        filter_valid_segment_pages.for_training([])
        filter_valid_segment_pages.for_prediction([])

        prediction_data = self.get_prediction_data(page_numbers=[2])
        pages_per_document = filter_valid_segment_pages.for_prediction([prediction_data])
        self.assertEqual(1, len(pages_per_document))
        self.assertEqual([1, 2], pages_per_document[0])

    def test_get_valid_pages_for_prediction_empty(self):
        tenant = "tenant_filter"
        extraction_id = "property_filter"

        shutil.rmtree(join(DATA_FOLDER_PATH, tenant), ignore_errors=True)

        filter_valid_segment_pages = FilterValidSegmentsPages(
            ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id)
        )
        labeled_data_json_path = Path(filter_valid_segment_pages.labeled_data_json_path)
        os.makedirs(labeled_data_json_path.parent, exist_ok=True)
        labeled_data_json_path.write_text("")

        prediction_data = self.get_prediction_data(page_numbers=[2])
        pages_per_document = filter_valid_segment_pages.for_prediction([prediction_data])
        self.assertEqual(1, len(pages_per_document))
        self.assertEqual([1, 2], pages_per_document[0])

    def test_get_valid_pages_only_prediction(self):
        tenant = "tenant_filter"
        extraction_id = "property_filter"

        shutil.rmtree(join(DATA_FOLDER_PATH, tenant), ignore_errors=True)

        filter_valid_segment_pages = FilterValidSegmentsPages(
            ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id)
        )
        prediction_data = self.get_prediction_data(page_numbers=[2])
        pages_per_document = filter_valid_segment_pages.for_prediction([prediction_data])
        self.assertEqual(1, len(pages_per_document))
        self.assertEqual([1, 2], pages_per_document[0])

    def test_get_valid_pages_prediction_first_page(self):
        tenant = "tenant_filter"
        extraction_id = "property_filter"

        shutil.rmtree(join(DATA_FOLDER_PATH, tenant), ignore_errors=True)

        filter_valid_segment_pages = FilterValidSegmentsPages(
            ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id)
        )
        labeled_data = self.get_labeled_data(labels_page_numbers=[1])
        filter_valid_segment_pages.for_training([labeled_data])
        prediction_data_1 = self.get_prediction_data(page_numbers=[20])
        pages_per_document = filter_valid_segment_pages.for_prediction([prediction_data_1])
        self.assertEqual(1, len(pages_per_document))
        self.assertEqual([1, 2], pages_per_document[0])

    def test_get_first_four_and_last_four_when_no_labels(self):
        tenant = "tenant_filter"
        extraction_id = "property_filter"

        shutil.rmtree(join(DATA_FOLDER_PATH, tenant), ignore_errors=True)

        filter_valid_segment_pages = FilterValidSegmentsPages(
            ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id)
        )

        labeled_data_1 = self.get_labeled_data(labels_page_numbers=[], number_of_pages=3)
        labeled_data_2 = self.get_labeled_data(labels_page_numbers=[], number_of_pages=8)
        labeled_data_3 = self.get_labeled_data(labels_page_numbers=[], number_of_pages=9)
        labeled_data_4 = self.get_labeled_data(labels_page_numbers=[], number_of_pages=50)

        training_data = [labeled_data_1, labeled_data_2, labeled_data_3, labeled_data_4]
        training_pages_per_document = filter_valid_segment_pages.for_training(training_data)

        prediction_data_1 = self.get_prediction_data(page_numbers=list(range(1, 4)))
        prediction_data_2 = self.get_prediction_data(page_numbers=list(range(1, 9)))
        prediction_data_3 = self.get_prediction_data(page_numbers=list(range(1, 10)))
        prediction_data_4 = self.get_prediction_data(page_numbers=list(range(1, 51)))
        prediction_data = [prediction_data_1, prediction_data_2, prediction_data_3, prediction_data_4]

        prediction_pages_per_document = filter_valid_segment_pages.for_prediction(prediction_data)

        self.assertEqual([1, 2, 3], training_pages_per_document[0])
        self.assertEqual([1, 2, 3], prediction_pages_per_document[0])

        self.assertEqual([1, 2, 3, 4, 5, 6, 7, 8], training_pages_per_document[1])
        self.assertEqual([1, 2, 3, 4, 5, 6, 7, 8], prediction_pages_per_document[1])

        self.assertEqual([1, 2, 3, 4, 6, 7, 8, 9], training_pages_per_document[2])
        self.assertEqual([1, 2, 3, 4, 6, 7, 8, 9], prediction_pages_per_document[2])

        self.assertEqual([1, 2, 3, 4, 47, 48, 49, 50], training_pages_per_document[3])
        self.assertEqual([1, 2, 3, 4, 47, 48, 49, 50], prediction_pages_per_document[3])
