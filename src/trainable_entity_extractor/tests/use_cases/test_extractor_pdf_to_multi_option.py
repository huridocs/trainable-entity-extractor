import shutil
from os.path import join
from unittest import TestCase

from trainable_entity_extractor.adapters.ExtractorLogger import ExtractorLogger
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.PdfToMultiOptionExtractor import (
    PdfToMultiOptionExtractor,
)
from trainable_entity_extractor.adapters.LocalExtractionDataRetriever import LocalExtractionDataRetriever
from trainable_entity_extractor.adapters.LocalModelStorage import LocalModelStorage
from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.domain.XmlFile import XmlFile
from trainable_entity_extractor.config import APP_PATH, CACHE_PATH
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.SegmentationData import SegmentationData
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.TrainUseCase import TrainUseCase
from trainable_entity_extractor.use_cases.PredictUseCase import PredictUseCase

extraction_id = "test_pdf_to_multi_option"
extraction_identifier = ExtractionIdentifier(run_name="unit_test", extraction_name=extraction_id)
TEST_XML_PATH = APP_PATH / "trainable_entity_extractor" / "tests" / "test_files"


class TestExtractorPdfToMultiOption(TestCase):
    def setUp(self):
        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)
        self.data_retriever = LocalExtractionDataRetriever()
        self.model_storage = LocalModelStorage()
        self.extractors = [PdfToMultiOptionExtractor]
        logger = ExtractorLogger()
        self.train_use_case = TrainUseCase(extractors=self.extractors, logger=logger)
        self.predict_use_case = PredictUseCase(extractors=self.extractors, logger=logger)

    def tearDown(self):
        # shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)
        shutil.rmtree(CACHE_PATH, ignore_errors=True)

    def _create_and_train_model(self, method_name: str, extraction_data: ExtractionData) -> TrainableEntityExtractorJob:
        # Save extraction data
        self.data_retriever.save_extraction_data(extraction_identifier, extraction_data)

        # Get available jobs for training
        jobs = self.train_use_case.get_jobs(extraction_data)
        self.assertGreater(len(jobs), 0, "No training jobs available")

        # Find the specific extractor job by method name
        extractor_job = None
        for job in jobs:
            if job.method_name == method_name:
                extractor_job = job
                break

        self.assertIsNotNone(extractor_job, f"Method {method_name} not found in available jobs")

        # Train the model
        success, message = self.train_use_case.train_one_method(extractor_job, extraction_data)
        self.assertTrue(success, f"Training failed: {message}")

        # Save the trained job
        self.model_storage.upload_model(extraction_identifier=extraction_identifier, extractor_job=extractor_job)

        return extractor_job

    def test_get_pdf_multi_option_suggestions(self):
        options = [Option(id=f"id{n}", label=str(n) + " February 2021") for n in range(16)]

        segmentation_data = SegmentationData(page_width=612, page_height=792, xml_segments_boxes=[], label_segments_boxes=[])

        test_xml_path = join(TEST_XML_PATH, "test.xml")
        xml_file = XmlFile(extraction_identifier=extraction_identifier, to_train=True, xml_file_name=test_xml_path)
        pdf_data = PdfData.from_xml_file(xml_file, segmentation_data)
        labeled_data = LabeledData(values=[Option(id="id15", label="15")], xml_file_name="test.xml", id=extraction_id)
        samples = [TrainingSample(pdf_data=pdf_data, labeled_data=labeled_data)]
        extraction_data = ExtractionData(
            samples=samples, extraction_identifier=extraction_identifier, multi_value=True, options=options
        )

        method_name = "FuzzyFirst"
        extractor_job = self._create_and_train_model(method_name, extraction_data)

        # Create prediction samples
        prediction_samples = [PredictionSample(pdf_data=pdf_data, entity_name="test.xml")]
        self.data_retriever.save_prediction_data(extraction_identifier, prediction_samples)

        # Make predictions
        suggestions = self.predict_use_case.predict(extractor_job, prediction_samples)

        self.assertEqual(1, len(suggestions))
        self.assertEqual(extraction_id, suggestions[0].id)
        self.assertEqual("test.xml", suggestions[0].xml_file_name)
        segment_text = '<p class="ix_matching_paragraph"><span class="ix_match">15 February 2021</span></p>'
        self.assertEqual([Value(id="id15", label="15 February 2021", segment_text=segment_text)], suggestions[0].values)
