from os.path import join
from unittest import TestCase

from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyFirst import (
    FuzzyFirst,
)
from trainable_entity_extractor.config import APP_PATH
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.SegmentationData import SegmentationData
from trainable_entity_extractor.domain.XmlFile import XmlFile

TEST_XML_PATH = APP_PATH / "trainable_entity_extractor" / "tests" / "test_files"


class TestFuzzyFirstXml(TestCase):
    TENANT = "unit_test"
    extraction_id = "FuzzyFirstXml"

    def test_fuzzy_first_finds_option_in_xml(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        segmentation_data = SegmentationData(page_width=612, page_height=792, xml_segments_boxes=[], label_segments_boxes=[])
        xml_file = XmlFile(
            extraction_identifier=extraction_identifier, to_train=True, xml_file_name=str(join(TEST_XML_PATH, "test.xml"))
        )
        pdf_data = PdfData.from_xml_file(xml_file, segmentation_data)

        options = [Option(id="1", label="Assembly"), Option(id="2", label="Nations")]
        prediction_sample = PredictionSample(pdf_data=pdf_data, entity_name="test.xml")
        prediction_samples_data = PredictionSamplesData(
            multi_value=False, options=options, prediction_samples=[prediction_sample]
        )
        predictions = FuzzyFirst().set_extraction_identifier(extraction_identifier).predict(prediction_samples_data)

        self.assertEqual(1, len(predictions))
        self.assertEqual(1, len(predictions[0]))
        self.assertEqual("2", predictions[0][0].id)
        self.assertEqual("Nations", predictions[0][0].label)
        self.assertEqual("United Nations", predictions[0][0].segment_text)
