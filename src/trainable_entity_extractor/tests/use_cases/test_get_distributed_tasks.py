import shutil
from unittest import TestCase

from trainable_entity_extractor.adapters.ExtractorLogger import ExtractorLogger
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.ports.Logger import Logger
from trainable_entity_extractor.use_cases.TrainUseCase import TrainUseCase
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.TextToTextExtractor import TextToTextExtractor
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.TextToMultiOptionExtractor import (
    TextToMultiOptionExtractor,
)

extraction_id = "test_get_distributed_jobs"
extraction_identifier = ExtractionIdentifier(extraction_name=extraction_id)


class TestGetDistributedJobs(TestCase):
    def setUp(self):
        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)
        self.logger = ExtractorLogger()
        self.train_use_case = None

    def tearDown(self):
        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)

    @staticmethod
    def create_text_extraction_data(samples_count=5) -> ExtractionData:
        samples = []
        for i in range(samples_count):
            labeled_data = LabeledData(
                tenant=f"tenant_{i}",
                id=f"id_{i}",
                xml_file_name=f"file_{i}.xml",
                entity_name="test_entity",
                language_iso="en",
                label_text=f"Extracted text {i}",
                source_text=f"This is source text for extraction {i}. Date: 2023-{i+1:02d}-15",
                values=[],
            )

            training_sample = TrainingSample(pdf_data=None, labeled_data=labeled_data)
            samples.append(training_sample)

        return ExtractionData(samples=samples, options=[], multi_value=False, extraction_identifier=extraction_identifier)

    @staticmethod
    def create_multi_option_extraction_data(samples_count=5) -> ExtractionData:
        samples = []
        for i in range(samples_count):
            labeled_data = LabeledData(
                tenant=f"tenant_{i}",
                id=f"id_{i}",
                xml_file_name=f"file_{i}.xml",
                entity_name="test_entity",
                language_iso="en",
                label_text=f"Sample text content {i}",
                source_text=f"This is sample text content for testing {i}",
                values=[Option(id=f"opt_{i}", label=f"Option {i}")],
            )

            training_sample = TrainingSample(pdf_data=None, labeled_data=labeled_data)
            samples.append(training_sample)

        return ExtractionData(
            samples=samples,
            options=[Option(id="opt_1", label="Option 1"), Option(id="opt_2", label="Option 2")],
            multi_value=False,
            extraction_identifier=extraction_identifier,
        )

    def test_get_distributed_jobs_with_text_to_text_extractor(self):
        extraction_data = self.create_text_extraction_data()
        self.train_use_case = TrainUseCase(extractors=[TextToTextExtractor], logger=self.logger)

        jobs = self.train_use_case.get_jobs(extraction_data)

        self.assertIsInstance(jobs, list)
        self.assertGreater(len(jobs), 0)
        for job in jobs:
            self.assertIsInstance(job, TrainableEntityExtractorJob)
            self.assertEqual(job.extractor_name, "TextToTextExtractor")
            self.assertEqual(job.run_name, extraction_data.extraction_identifier.run_name)
            self.assertEqual(job.extraction_name, extraction_data.extraction_identifier.extraction_name)

    def test_get_distributed_jobs_with_multi_option_extractor(self):
        extraction_data = self.create_multi_option_extraction_data()
        self.train_use_case = TrainUseCase(extractors=[TextToMultiOptionExtractor], logger=self.logger)

        jobs = self.train_use_case.get_jobs(extraction_data)

        self.assertIsInstance(jobs, list)
        self.assertGreater(len(jobs), 0)
        for job in jobs:
            self.assertIsInstance(job, TrainableEntityExtractorJob)
            self.assertEqual(job.extractor_name, "TextToMultiOptionExtractor")

    def test_get_distributed_jobs_no_compatible_extractor(self):
        extraction_data = ExtractionData(
            samples=[], options=[], multi_value=False, extraction_identifier=extraction_identifier
        )
        self.train_use_case = TrainUseCase(extractors=[TextToTextExtractor], logger=self.logger)

        jobs = self.train_use_case.get_jobs(extraction_data)

        self.assertEqual(len(jobs), 0)

    def test_get_distributed_jobs_with_incompatible_data_for_text_extractor(self):
        extraction_data = ExtractionData(
            samples=[],
            options=[Option(id="opt_1", label="Option 1")],
            multi_value=False,
            extraction_identifier=extraction_identifier,
        )
        self.train_use_case = TrainUseCase(extractors=[TextToTextExtractor], logger=self.logger)

        jobs = self.train_use_case.get_jobs(extraction_data)

        self.assertEqual(len(jobs), 0)

    def test_get_distributed_jobs_returns_multiple_jobs_from_single_extractor(self):
        extraction_data = self.create_text_extraction_data()
        self.train_use_case = TrainUseCase(extractors=[TextToTextExtractor], logger=self.logger)

        jobs = self.train_use_case.get_jobs(extraction_data)

        self.assertIsInstance(jobs, list)
        self.assertGreater(len(jobs), 1)

        method_names = [job.method_name for job in jobs]
        unique_method_names = set(method_names)
        self.assertEqual(len(method_names), len(unique_method_names))

    def test_get_distributed_jobs_with_minimal_valid_data(self):
        samples = []
        labeled_data = LabeledData(
            tenant="test_tenant",
            id="test_id",
            xml_file_name="test_file.xml",
            entity_name="test_entity",
            language_iso="en",
            label_text="Test label",
            source_text="Test source text",
            values=[],
        )
        training_sample = TrainingSample(pdf_data=None, labeled_data=labeled_data)
        samples.append(training_sample)

        extraction_data = ExtractionData(
            samples=samples, options=[], multi_value=False, extraction_identifier=extraction_identifier
        )
        self.train_use_case = TrainUseCase(extractors=[TextToTextExtractor], logger=self.logger)

        jobs = self.train_use_case.get_jobs(extraction_data)

        self.assertIsInstance(jobs, list)
        self.assertGreater(len(jobs), 0)

    def test_get_distributed_jobs_validates_job_properties(self):
        extraction_data = self.create_text_extraction_data()
        self.train_use_case = TrainUseCase(extractors=[TextToTextExtractor], logger=self.logger)

        jobs = self.train_use_case.get_jobs(extraction_data)

        for job in jobs:
            self.assertIsInstance(job.method_name, str)
            self.assertGreater(len(job.method_name), 0)
            self.assertIsInstance(job.gpu_needed, bool)
            self.assertIsInstance(job.timeout, int)
            self.assertGreater(job.timeout, 0)
            self.assertIsInstance(job.options, list)
            self.assertIsInstance(job.multi_value, bool)
