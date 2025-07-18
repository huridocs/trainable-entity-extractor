from os.path import join
from unittest import TestCase

from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.use_cases.TrainableEntityExtractor import TrainableEntityExtractor
from trainable_entity_extractor.use_cases.XmlFile import XmlFile
from trainable_entity_extractor.config import APP_PATH
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.SegmentationData import SegmentationData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample

extraction_id = "test_pdf_to_multi_option"
extraction_identifier = ExtractionIdentifier(run_name="unit_test", extraction_name=extraction_id)
TEST_XML_PATH = APP_PATH / "trainable_entity_extractor" / "tests" / "test_files"


class TestExtractorPdfToMultiOption(TestCase):
    def test_get_pdf_multi_option_suggestions(self):
        options = [Option(id=f"id{n}", label=str(n)) for n in range(16)]

        segmentation_data = SegmentationData(page_width=612, page_height=792, xml_segments_boxes=[], label_segments_boxes=[])

        test_xml_path = join(TEST_XML_PATH, "test.xml")
        xml_file = XmlFile(extraction_identifier=extraction_identifier, to_train=True, xml_file_name=test_xml_path)
        pdf_data = PdfData.from_xml_file(xml_file, segmentation_data)
        labeled_data = LabeledData(values=[Option(id="id15", label="15")], xml_file_name="test.xml", id=extraction_id)
        samples = [TrainingSample(pdf_data=pdf_data, labeled_data=labeled_data)]
        extraction_data = ExtractionData(
            samples=samples, extraction_identifier=extraction_identifier, multi_value=True, options=options
        )

        trainable_entity_extractor = TrainableEntityExtractor(extraction_identifier=extraction_identifier)
        trainable_entity_extractor.train(extraction_data)

        samples = [PredictionSample(pdf_data=pdf_data, entity_name="test.xml")]
        suggestions = trainable_entity_extractor.predict(samples)

        self.assertEqual(1, len(suggestions))
        self.assertEqual(extraction_id, suggestions[0].id)
        self.assertEqual("test.xml", suggestions[0].xml_file_name)
        segment_text = "A ________ United Nations ________ /INF/76/1 ________ General Assembly ________ Distr.: General ________ 15 February 2021 ________ Original: English ________ Seventy-sixth session ________ Opening dates of forthcoming regular sessions of the ________ General Assembly and of the general debate ________ Note by the Secretariat ________ 1. ________ In paragraph 1 of its resolution ________ 57/301 ________ , the General Assembly decided to amend ________ rule 1 of its rules of procedure to read: “The General Assembly shall meet every year ________ in regular session commencing on the Tuesday of the third week in September, ________ counting from the first week that contains at least one working day.” In paragraph 2 ________ of the resolution, the Assembly also decided that the general debate in the Assembly"
        self.assertEqual([Value(id="id15", label="15", segment_text=segment_text)], suggestions[0].values)
