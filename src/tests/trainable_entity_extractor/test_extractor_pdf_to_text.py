import unittest
from os.path import join

from unittest import TestCase

from pdf_token_type_labels.TokenType import TokenType

from trainable_entity_extractor.XmlFile import XmlFile
from trainable_entity_extractor.config import APP_PATH
from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.LabeledData import LabeledData
from trainable_entity_extractor.data.PdfData import PdfData
from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.data.SegmentBox import SegmentBox
from trainable_entity_extractor.data.SegmentationData import SegmentationData
from trainable_entity_extractor.TrainableEntityExtractor import TrainableEntityExtractor
from trainable_entity_extractor.data.TrainingSample import TrainingSample

extraction_id = "test_pdf_to_text"
extraction_identifier = ExtractionIdentifier(extraction_name=extraction_id)
TEST_XML_PATH = f"{APP_PATH}/tests/trainable_entity_extractor/test_files"


class TestExtractorPdfToText(TestCase):
    def test_create_model_should_do_nothing_when_no_xml(self):
        segment_box = SegmentBox(
            left=125,
            top=247,
            width=319,
            height=29,
            page_width=612,
            page_height=792,
            page_number=1,
            segment_type=TokenType.TEXT,
        )
        labeled_data = LabeledData(label_text="text", label_segments_boxes=[segment_box])

        pdf_data = PdfData.from_texts(["text"])
        samples = [TrainingSample(pdf_data=pdf_data, labeled_data=labeled_data)]

        extraction_data = ExtractionData(samples=samples, extraction_identifier=extraction_identifier)

        trainable_entity_extractor = TrainableEntityExtractor(extraction_identifier=extraction_identifier)
        success, error = trainable_entity_extractor.train(extraction_data)
        self.assertEqual(success, True)

    def test_create_model_when_blank_document(self):
        segment_box = SegmentBox(
            left=123,
            top=48,
            width=83,
            height=12,
            page_width=612,
            page_height=792,
            page_number=1,
            segment_type=TokenType.TEXT,
        )
        labeled_data = LabeledData(label_text="some text", label_segments_boxes=[segment_box])
        segmentation_data = SegmentationData(
            page_width=612, page_height=792, xml_segments_boxes=[], label_segments_boxes=[segment_box]
        )

        blank_xml = join(TEST_XML_PATH, "blank.xml")
        xml_file = XmlFile(extraction_identifier=extraction_identifier, to_train=True, xml_file_name=blank_xml)
        pdf_data = PdfData.from_xml_file(xml_file, segmentation_data)
        samples = [TrainingSample(pdf_data=pdf_data, labeled_data=labeled_data)]

        extraction_data = ExtractionData(samples=samples, extraction_identifier=extraction_identifier)

        trainable_entity_extractor = TrainableEntityExtractor(extraction_identifier=extraction_identifier)
        success, error = trainable_entity_extractor.train(extraction_data)
        self.assertEqual(success, True)

    def test_calculate_suggestions(self):
        segment_box = SegmentBox(
            left=400,
            top=115,
            width=74,
            height=9,
            page_width=612,
            page_height=792,
            page_number=1,
            segment_type=TokenType.TEXT,
        )

        labeled_data = LabeledData(label_text="Original: English", label_segments_boxes=[segment_box])
        segmentation_data = SegmentationData(
            page_width=612, page_height=792, xml_segments_boxes=[], label_segments_boxes=[segment_box]
        )

        test_xml = join(TEST_XML_PATH, "test.xml")
        xml_file = XmlFile(extraction_identifier=extraction_identifier, to_train=True, xml_file_name=test_xml)
        pdf_data = PdfData.from_xml_file(xml_file, segmentation_data)
        samples = [TrainingSample(pdf_data=pdf_data, labeled_data=labeled_data)] * 7

        extraction_data = ExtractionData(samples=samples, extraction_identifier=extraction_identifier)

        trainable_entity_extractor = TrainableEntityExtractor(extraction_identifier=extraction_identifier)
        trainable_entity_extractor.train(extraction_data)

        samples = [PredictionSample(pdf_data=pdf_data, entity_name="test.xml")]
        suggestion = trainable_entity_extractor.predict(samples)[0]

        self.assertEqual(extraction_id, suggestion.id)
        self.assertEqual("test.xml", suggestion.xml_file_name)
        self.assertTrue("Original: English" in suggestion.segment_text)
        self.assertEqual("Original: English", suggestion.text)
        self.assertEqual(1, suggestion.page_number)

        self.assertEqual(len(suggestion.segments_boxes), 2)
        self.assertEqual(397.0, suggestion.segments_boxes[0].left)
        self.assertEqual(90.0, suggestion.segments_boxes[0].top)
        self.assertEqual(1, suggestion.segments_boxes[0].page_number)

    @unittest.SkipTest
    def test_get_semantic_suggestions(self):
        segment_box = SegmentBox(left=397, top=115, page_width=612, page_height=792, width=74, height=9, page_number=1)

        labeled_data = LabeledData(label_text="English1", language_iso="en", label_segments_boxes=[segment_box])
        segmentation_data = SegmentationData(
            page_width=612, page_height=792, xml_segments_boxes=[], label_segments_boxes=[segment_box]
        )

        test_xml = join(TEST_XML_PATH, "test.xml")
        xml_file = XmlFile(extraction_identifier=extraction_identifier, to_train=True, xml_file_name=test_xml)
        pdf_data = PdfData.from_xml_file(xml_file, segmentation_data)
        samples = [TrainingSample(pdf_data=pdf_data, labeled_data=labeled_data)] * 7

        extraction_data = ExtractionData(samples=samples, extraction_identifier=extraction_identifier)

        trainable_entity_extractor = TrainableEntityExtractor(extraction_identifier=extraction_identifier)
        task_calculated, error_message = trainable_entity_extractor.train(extraction_data)

        samples = [PredictionSample(pdf_data=pdf_data, entity_name="test.xml")]
        suggestion = trainable_entity_extractor.predict(samples)[0]

        self.assertTrue(task_calculated)
        self.assertEqual("default", suggestion.tenant)
        self.assertEqual(extraction_id, suggestion.id)
        self.assertEqual("test.xml", suggestion.xml_file_name)
        self.assertEqual("Original: English", suggestion.segment_text)
        self.assertEqual("English1", suggestion.text)

        self.assertEqual(1, len(suggestion.segments_boxes))
        self.assertEqual(397.0, suggestion.segments_boxes[0].left)
        self.assertEqual(114.0, suggestion.segments_boxes[0].top)
        self.assertEqual(77.0, suggestion.segments_boxes[0].width)
        self.assertEqual(11.0, suggestion.segments_boxes[0].height)
        self.assertEqual(1, suggestion.segments_boxes[0].page_number)

    def test_get_semantic_suggestions_numeric(self):
        segment_box = SegmentBox(left=397, top=91, page_width=612, page_height=792, width=10, height=9, page_number=1)

        labeled_data = LabeledData(label_text="15", language_iso="en", label_segments_boxes=[segment_box])
        segmentation_data = SegmentationData(
            page_width=612, page_height=792, xml_segments_boxes=[], label_segments_boxes=[segment_box]
        )

        test_xml = join(TEST_XML_PATH, "test.xml")
        xml_file = XmlFile(extraction_identifier=extraction_identifier, to_train=True, xml_file_name=test_xml)
        pdf_data = PdfData.from_xml_file(xml_file, segmentation_data)
        samples = [TrainingSample(pdf_data=pdf_data, labeled_data=labeled_data)] * 7

        extraction_data = ExtractionData(samples=samples, extraction_identifier=extraction_identifier)

        trainable_entity_extractor = TrainableEntityExtractor(extraction_identifier=extraction_identifier)
        task_calculated, error_message = trainable_entity_extractor.train(extraction_data)

        samples = [PredictionSample(pdf_data=pdf_data, entity_name="test.xml")]
        suggestion = trainable_entity_extractor.predict(samples)[0]

        self.assertTrue(task_calculated)
        self.assertEqual("default", suggestion.tenant)
        self.assertEqual(extraction_id, suggestion.id)
        self.assertEqual("test.xml", suggestion.xml_file_name)
        self.assertTrue("15 February 2021" in suggestion.segment_text)
        self.assertEqual("15", suggestion.text)

    def test_get_suggestions_blank_document(self):
        segment_box = SegmentBox(left=397, top=91, page_width=612, page_height=792, width=10, height=9, page_number=1)

        labeled_data = LabeledData(label_text="15", language_iso="en", label_segments_boxes=[segment_box])
        segmentation_data = SegmentationData(
            page_width=612, page_height=792, xml_segments_boxes=[], label_segments_boxes=[segment_box]
        )

        test_xml = join(TEST_XML_PATH, "test.xml")
        xml_file = XmlFile(extraction_identifier=extraction_identifier, to_train=True, xml_file_name=test_xml)
        pdf_data = PdfData.from_xml_file(xml_file, segmentation_data)
        samples = [TrainingSample(pdf_data=pdf_data, labeled_data=labeled_data)] * 7

        extraction_data = ExtractionData(samples=samples, extraction_identifier=extraction_identifier)

        trainable_entity_extractor = TrainableEntityExtractor(extraction_identifier=extraction_identifier)
        task_calculated, error_message = trainable_entity_extractor.train(extraction_data)

        blank_xml = join(TEST_XML_PATH, "blank.xml")
        xml_file = XmlFile(extraction_identifier=extraction_identifier, to_train=False, xml_file_name=blank_xml)
        pdf_data = PdfData.from_xml_file(xml_file, segmentation_data)

        samples = [PredictionSample(pdf_data=pdf_data, entity_name="blank.xml")]
        suggestion = trainable_entity_extractor.predict(samples)[0]

        self.assertTrue(task_calculated)
        self.assertEqual("default", suggestion.tenant)
        self.assertEqual(extraction_id, suggestion.id)
        self.assertEqual("blank.xml", suggestion.xml_file_name)
        self.assertEqual("", suggestion.segment_text)
        self.assertEqual("", suggestion.text)

    def test_get_suggestions_no_pages_document(self):
        segment_box = SegmentBox(left=397, top=91, page_width=612, page_height=792, width=10, height=9, page_number=1)

        labeled_data = LabeledData(label_text="15", language_iso="en", label_segments_boxes=[segment_box])
        segmentation_data = SegmentationData(
            page_width=612, page_height=792, xml_segments_boxes=[], label_segments_boxes=[segment_box]
        )

        test_xml = join(TEST_XML_PATH, "test.xml")
        xml_file = XmlFile(extraction_identifier=extraction_identifier, to_train=True, xml_file_name=test_xml)
        pdf_data = PdfData.from_xml_file(xml_file, segmentation_data)
        samples = [TrainingSample(pdf_data=pdf_data, labeled_data=labeled_data)] * 7

        extraction_data = ExtractionData(samples=samples, extraction_identifier=extraction_identifier)

        trainable_entity_extractor = TrainableEntityExtractor(extraction_identifier=extraction_identifier)
        task_calculated, error_message = trainable_entity_extractor.train(extraction_data)

        no_pages_xml = join(TEST_XML_PATH, "no_pages.xml")
        xml_file = XmlFile(extraction_identifier=extraction_identifier, to_train=False, xml_file_name=no_pages_xml)
        pdf_data = PdfData.from_xml_file(xml_file, segmentation_data)

        samples = [PredictionSample(pdf_data=pdf_data, entity_name="no_pages.xml")]
        suggestion = trainable_entity_extractor.predict(samples)[0]

        self.assertTrue(task_calculated)
        self.assertEqual("default", suggestion.tenant)
        self.assertEqual(extraction_id, suggestion.id)
        self.assertEqual("no_pages.xml", suggestion.xml_file_name)
        self.assertEqual("", suggestion.segment_text)
        self.assertEqual("", suggestion.text)
