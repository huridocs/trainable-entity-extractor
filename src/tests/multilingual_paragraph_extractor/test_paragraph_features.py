from pathlib import Path
from unittest import TestCase

from pdf_token_type_labels.TokenType import TokenType

from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures
from trainable_entity_extractor.XmlFile import XmlFile
from trainable_entity_extractor.config import ROOT_PATH, APP_PATH
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.PdfData import PdfData
from trainable_entity_extractor.data.SegmentBox import SegmentBox
from trainable_entity_extractor.data.SegmentationData import SegmentationData


class TestParagraphFeatures(TestCase):
    identifier = ExtractionIdentifier(run_name="paragraph", extraction_name="id")

    def test_extract_paragraphs(self):
        xml_path = Path(APP_PATH, "tests", "multilingual_paragraph_extractor", "resources", "test.xml")

        with open(xml_path, "rb") as file:
            xml_file = XmlFile(extraction_identifier=self.identifier, to_train=True, xml_file_name="test.xml")
            xml_file.save(file_content=file.read())

        segmentation_data = SegmentationData(
            page_width=612,
            page_height=792,
            xml_segments_boxes=[
                SegmentBox(
                    left=397,
                    top=78,
                    width=79,
                    height=47,
                    page_number=1,
                    page_width=612,
                    page_height=792,
                    segment_type=TokenType.PAGE_HEADER,
                )
            ],
            label_segments_boxes=[],
        )
        pdf_data = PdfData.from_xml_file(xml_file=xml_file, segmentation_data=segmentation_data)

        paragraph: ParagraphFeatures = ParagraphFeatures.from_pdf_data(
            pdf_data=pdf_data, pdf_segment=pdf_data.pdf_data_segments[4]
        )

        self.assertEqual("Distr.: General 15 February 2021 Original: English", paragraph.text_cleaned)
        self.assertEqual(792, paragraph.page_height)
        self.assertEqual(612, paragraph.page_width)
        self.assertEqual(1, paragraph.page_number)

        self.assertEqual(397, paragraph.bounding_box.left)
        self.assertEqual(78, paragraph.bounding_box.top)
        self.assertEqual(75, paragraph.bounding_box.width)
        self.assertEqual(47, paragraph.bounding_box.height)

        self.assertEqual(4, paragraph.index)
        self.assertEqual(7, len(paragraph.words))
        self.assertEqual([15, 2021], paragraph.numbers)
        self.assertEqual([".", ":", ":"], paragraph.non_alphanumeric_characters)
        self.assertEqual("Distr.:", paragraph.first_word)

        self.assertEqual(10, paragraph.font.font_size)
        self.assertEqual(False, paragraph.font.bold)
        self.assertEqual(False, paragraph.font.italics)

    def test_extract_other_paragraphs(self):
        xml_path = Path(APP_PATH, "tests", "multilingual_paragraph_extractor", "resources", "test.xml")

        with open(xml_path, "rb") as file:
            xml_file = XmlFile(extraction_identifier=self.identifier, to_train=True, xml_file_name="test.xml")
            xml_file.save(file_content=file.read())

        segmentation_data = SegmentationData(
            page_width=612,
            page_height=792,
            xml_segments_boxes=[],
            label_segments_boxes=[],
        )
        pdf_data = PdfData.from_xml_file(xml_file=xml_file, segmentation_data=segmentation_data)

        paragraph: ParagraphFeatures = ParagraphFeatures.from_pdf_data(
            pdf_data=pdf_data, pdf_segment=pdf_data.pdf_data_segments[80]
        )

        self.assertEqual("2/2", paragraph.text_cleaned)
        self.assertEqual(792, paragraph.page_height)
        self.assertEqual(612, paragraph.page_width)
        self.assertEqual(2, paragraph.page_number)

        self.assertEqual(60, paragraph.bounding_box.left)
        self.assertEqual(749, paragraph.bounding_box.top)
        self.assertEqual(10, paragraph.bounding_box.width)
        self.assertEqual(8, paragraph.bounding_box.height)

        self.assertEqual(80, paragraph.index)
        self.assertEqual(1, len(paragraph.words))
        self.assertEqual([22], paragraph.numbers)
        self.assertEqual(["/"], paragraph.non_alphanumeric_characters)
        self.assertEqual("2/2", paragraph.first_word)

        self.assertEqual(9, paragraph.font.font_size)
        self.assertEqual(False, paragraph.font.bold)
        self.assertEqual(False, paragraph.font.italics)
