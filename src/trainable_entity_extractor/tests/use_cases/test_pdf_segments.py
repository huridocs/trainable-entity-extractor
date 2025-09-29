import shutil
from os.path import join
from unittest import TestCase

from trainable_entity_extractor.config import DATA_PATH, APP_PATH
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.SegmentBox import SegmentBox
from trainable_entity_extractor.domain.SegmentationData import SegmentationData
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.XmlFile import XmlFileUseCase
from pdf_token_type_labels.TokenType import TokenType


class TestPdfSegments(TestCase):
    TEST_XML_PATH = APP_PATH / "trainable_entity_extractor" / "tests" / "test_files"

    test_file_path = TEST_XML_PATH / "test.xml"
    no_pages_file_path = TEST_XML_PATH / "no_pages.xml"

    def test_get_pdf_features(self):
        tenant = "tenant_save"
        extraction_id = "property_save"

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

        segmentation_data = SegmentationData(
            page_width=612,
            page_height=792,
            xml_segments_boxes=[
                SegmentBox(
                    left=495.1,
                    top=42.6323,
                    width=56.96199999999999,
                    page_width=612,
                    page_height=792,
                    height=18.2164,
                    page_number=1,
                    type=TokenType.TEXT,
                ),
                SegmentBox(
                    left=123.38,
                    top=48.1103,
                    page_width=612,
                    page_height=792,
                    width=82.9812,
                    height=12.7624,
                    page_number=1,
                    type=TokenType.TEXT,
                ),
                SegmentBox(
                    left=123.38,
                    top=72.8529,
                    width=148.656,
                    height=17.895700000000005,
                    page_width=612,
                    page_height=792,
                    page_number=1,
                    type=TokenType.TEXT,
                ),
                SegmentBox(
                    left=123.38,
                    top=245.184,
                    width=317.406,
                    height=27.5377,
                    page_width=612,
                    page_height=792,
                    page_number=1,
                    type=TokenType.TEXT,
                ),
            ],
            label_segments_boxes=[
                SegmentBox(
                    left=125,
                    top=247,
                    width=319,
                    height=29,
                    page_width=612,
                    page_height=792,
                    page_number=1,
                    type=TokenType.TEXT,
                )
            ],
        )

        with open(self.test_file_path, "rb") as file:
            xml_file = XmlFileUseCase(
                extraction_identifier=ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id),
                to_train=True,
                xml_file_name="test.xml",
            )

            xml_file.save(file_content=file.read())

        pdf_segments = PdfData.from_xml_file(xml_file, segmentation_data, [])

        self.assertEqual(612, pdf_segments.pdf_features.pages[0].page_width)
        self.assertEqual(792, pdf_segments.pdf_features.pages[0].page_height)
        self.assertEqual(1, len([segment for segment in pdf_segments.pdf_data_segments if segment.ml_label == 1]))
        self.assertEqual("A/INF/76/1", pdf_segments.pdf_data_segments[0].text_content)
        self.assertEqual("United Nations", pdf_segments.pdf_data_segments[1].text_content)
        self.assertEqual("General Assembly", pdf_segments.pdf_data_segments[2].text_content)
        self.assertEqual(
            "Opening dates of forthcoming regular sessions of the General Assembly and of the general debate",
            [segment for segment in pdf_segments.pdf_data_segments if segment.ml_label == 1][0].text_content,
        )

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

    def test_get_pdf_features_when_no_pages(self):
        tenant = "tenant_save"
        extraction_id = "property_save"

        segmentation_data = SegmentationData(
            page_width=1,  # 612
            page_height=2,  # 396
            xml_segments_boxes=[],
            label_segments_boxes=[],
        )

        with open(self.no_pages_file_path, "rb") as file:
            xml_file = XmlFileUseCase(
                extraction_identifier=ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id),
                to_train=True,
                xml_file_name="no_pages.xml",
            )

            xml_file.save(file_content=file.read())

        pdf_features = PdfData.from_xml_file(xml_file, segmentation_data, [])

        self.assertEqual(0, len(pdf_features.pdf_data_segments))

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

    def test_get_pdf_features_when_no_file(self):
        tenant = "tenant_save"
        extraction_id = "property_save"

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)
        segmentation_data = SegmentationData(
            page_width=1,  # 612
            page_height=2,  # 396
            xml_segments_boxes=[],
            label_segments_boxes=[],
        )

        xml_file = XmlFileUseCase(
            extraction_identifier=ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id),
            to_train=True,
            xml_file_name="test.xml",
        )

        pdf_segments = PdfData.from_xml_file(xml_file, segmentation_data, [])

        self.assertEqual(0, len(pdf_segments.pdf_data_segments))

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

    def test_get_pdf_features_should_be_empty_when_no_file_because_different_extraction_id(
        self,
    ):
        tenant = "tenant_save"
        extraction_id = "property_save"

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

        segmentation_data = SegmentationData(
            page_width=1,  # 612
            page_height=2,  # 396
            xml_segments_boxes=[
                SegmentBox(left=0, top=0.3282828, width=1, height=0.1767676, page_width=612, page_height=792, page_number=2),
            ],
            label_segments_boxes=[
                SegmentBox(
                    left=0.49019,
                    top=0.37878,
                    width=0.008169,
                    height=0.0126,
                    page_width=612,
                    page_height=792,
                    page_number=2,
                )
            ],
        )
        with open(self.test_file_path, "rb") as file:
            XmlFileUseCase(
                extraction_identifier=ExtractionIdentifier(run_name=tenant, extraction_name="different_extraction_id"),
                to_train=False,
                xml_file_name="test.xml",
            ).save(file_content=file.read())

        xml_file = XmlFileUseCase(
            extraction_identifier=ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id),
            to_train=False,
            xml_file_name="test.xml",
        )

        pdf_features = PdfData.from_xml_file(xml_file, segmentation_data, [])

        self.assertEqual(0, len(pdf_features.pdf_data_segments))

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

    def test_filter_valid_segment_pages(self):
        tenant = "tenant_save"
        extraction_id = "property_save"

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

        segmentation_data = SegmentationData(
            page_width=612,
            page_height=792,
            xml_segments_boxes=[
                SegmentBox(
                    left=1, top=1, width=56.96199999999999, height=18.2164, page_width=612, page_height=792, page_number=1
                )
            ],
            label_segments_boxes=[
                SegmentBox(left=125, top=247, width=319, height=29, page_width=612, page_height=792, page_number=1)
            ],
        )

        with open(self.test_file_path, "rb") as file:
            xml_file = XmlFileUseCase(
                extraction_identifier=ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id),
                to_train=True,
                xml_file_name="test.xml",
            )

            xml_file.save(file_content=file.read())

        pdf_segments = PdfData.from_xml_file(xml_file, segmentation_data, [1])

        self.assertEqual(0, len([segment for segment in pdf_segments.pdf_data_segments if segment.page_number > 1]))

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)
