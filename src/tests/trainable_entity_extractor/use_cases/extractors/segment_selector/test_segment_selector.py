import json
import shutil
from os import makedirs
from os.path import exists, join
from time import time
from unittest import TestCase

from trainable_entity_extractor.config import APP_PATH, DATA_PATH
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.SegmentBox import SegmentBox
from trainable_entity_extractor.domain.SegmentationData import SegmentationData
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.use_cases.XmlFile import XmlFile
from trainable_entity_extractor.use_cases.extractors.segment_selector.SegmentSelector import SegmentSelector
from pdf_token_type_labels.TokenType import TokenType


class TestSegmentSelector(TestCase):
    TENANT = "segment_selector_test"
    extraction_id = "extraction_id"
    EXTRACTION_IDENTIFIER = ExtractionIdentifier(run_name=TENANT, extraction_name=extraction_id)
    TEST_XML_NAME = "test.xml"

    TEST_XML_PATH = join(APP_PATH, "tests", "trainable_entity_extractor", "test_files", TEST_XML_NAME)
    BASE_PATH = join(DATA_PATH, TENANT, extraction_id)

    labels = SegmentBox(
        left=400, top=115, width=74, height=9, page_number=1, page_width=612, page_height=792, segment_type=TokenType.TITLE
    )
    LABELED_DATA_JSON = {
        "tenant": TENANT,
        "id": extraction_id,
        "xml_file_name": TEST_XML_NAME,
        "language_iso": "en",
        "label_text": "text",
        "page_width": 612,
        "page_height": 792,
        "xml_segments_boxes": [],
        "label_segments_boxes": [json.loads(labels.model_dump_json())],
    }

    XML_FILE = XmlFile(
        extraction_identifier=EXTRACTION_IDENTIFIER,
        to_train=True,
        xml_file_name=TEST_XML_NAME,
    )

    def setUp(self):
        shutil.rmtree(join(DATA_PATH, TestSegmentSelector.TENANT), ignore_errors=True)

        makedirs(join(TestSegmentSelector.BASE_PATH, "xml_to_train"))
        test_folder_path = join(TestSegmentSelector.BASE_PATH, "xml_to_train", TestSegmentSelector.TEST_XML_NAME)
        shutil.copy(self.TEST_XML_PATH, test_folder_path)
        segment_selector = SegmentSelector(extraction_identifier=TestSegmentSelector.EXTRACTION_IDENTIFIER)
        segment_selector.prepare_model_folder()

    def tearDown(self):
        shutil.rmtree(join(DATA_PATH, TestSegmentSelector.TENANT), ignore_errors=True)

    def test_create_model(self):
        segmentation_data = SegmentationData.from_labeled_data(LabeledData(**TestSegmentSelector.LABELED_DATA_JSON))
        pdf_segments = PdfData.from_xml_file(TestSegmentSelector.XML_FILE, segmentation_data, [])

        segment_selector = SegmentSelector(extraction_identifier=TestSegmentSelector.EXTRACTION_IDENTIFIER)
        model_created, error = segment_selector.create_model(pdfs_data=[pdf_segments])

        self.assertTrue(model_created)
        self.assertEqual("", error)
        self.assertTrue(exists(join(TestSegmentSelector.BASE_PATH, "segment_predictor_model", "model.model")))
        self.assertFalse(exists(join(TestSegmentSelector.BASE_PATH, "multilingual_segment_predictor_model", "model.model")))

    def test_create_model_load_test(self):
        start = time()
        segmentation_data = SegmentationData.from_labeled_data(LabeledData(**TestSegmentSelector.LABELED_DATA_JSON))
        for i in range(20):
            PdfData.from_xml_file(TestSegmentSelector.XML_FILE, segmentation_data, [])

        print(time() - start, "create model")

    def test_set_extraction_segments(self):
        segmentation_data = SegmentationData.from_labeled_data(LabeledData(**TestSegmentSelector.LABELED_DATA_JSON))

        pdf_features = PdfData.from_xml_file(TestSegmentSelector.XML_FILE, segmentation_data, [])
        segment_selector = SegmentSelector(extraction_identifier=TestSegmentSelector.EXTRACTION_IDENTIFIER)
        segment_selector.prepare_model_folder()
        segment_selector.create_model(pdfs_data=[pdf_features])

        segment_selector = SegmentSelector(extraction_identifier=TestSegmentSelector.EXTRACTION_IDENTIFIER)
        segment_selector.set_extraction_segments(pdfs_data=[pdf_features])

        extraction_segments = [x for x in pdf_features.pdf_data_segments if x.ml_label]
        self.assertEqual(1, len(extraction_segments))
        self.assertEqual(1, extraction_segments[0].page_number)
        self.assertEqual("Original: English", extraction_segments[0].text_content)
