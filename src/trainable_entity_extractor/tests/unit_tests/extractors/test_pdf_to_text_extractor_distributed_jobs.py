import shutil
from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.ports.Logger import Logger
from trainable_entity_extractor.adapters.extractors.pdf_to_text_extractor.PdfToTextExtractor import PdfToTextExtractor

extraction_id = "test_pdf_to_text"
extraction_identifier = ExtractionIdentifier(extraction_name=extraction_id)


class TestLogger(Logger):
    def log(self, extraction_identifier: ExtractionIdentifier, message: str, severity: LogSeverity = LogSeverity.info, exception: Exception = None):
        pass


class TestPdfToTextExtractorDistributedJobs(TestCase):
    def setUp(self):
        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)
        self.logger = TestLogger()
        self.extractor = PdfToTextExtractor(extraction_identifier, self.logger)

    def tearDown(self):
        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)

    @staticmethod
    def create_sample_extraction_data(samples_count=5) -> ExtractionData:
        samples = []
        for i in range(samples_count):
            labeled_data = LabeledData(
                tenant=f"tenant_{i}",
                id=f"id_{i}",
                xml_file_name=f"file_{i}.xml",
                entity_name="test_entity",
                language_iso="en",
                label_text=f"Extracted text {i}",
                source_text=f"This is source text for PDF extraction {i}",
                values=[],
            )

            training_sample = TrainingSample(pdf_data=None, labeled_data=labeled_data)
            samples.append(training_sample)

        return ExtractionData(samples=samples, options=[], multi_value=False, extraction_identifier=extraction_identifier)

    def test_get_distributed_jobs_returns_jobs(self):
        extraction_data = self.create_sample_extraction_data()

        jobs = self.extractor.get_distributed_jobs(extraction_data)

        self.assertIsInstance(jobs, list)
        self.assertGreater(len(jobs), 0, "Should return at least one task")

    def test_get_distributed_jobs_creates_valid_task_objects(self):
        extraction_data = self.create_sample_extraction_data()

        jobs = self.extractor.get_distributed_jobs(extraction_data)

        for task in jobs:
            self.assertIsInstance(task, TrainableEntityExtractorJob)
            self.assertEqual(task.run_name, extraction_data.extraction_identifier.run_name)
            self.assertEqual(task.extraction_name, extraction_data.extraction_identifier.extraction_name)
            self.assertEqual(task.extractor_name, "PdfToTextExtractor")
            self.assertIsInstance(task.method_name, str)
            self.assertGreater(len(task.method_name), 0, "Method name should not be empty")
            self.assertIsInstance(task.gpu_needed, bool)
            self.assertIsInstance(task.timeout, int)
            self.assertGreater(task.timeout, 0, "Timeout should be positive")

    def test_get_distributed_jobs_includes_segment_selector_methods(self):
        extraction_data = self.create_sample_extraction_data()

        jobs = self.extractor.get_distributed_jobs(extraction_data)
        method_names = [task.method_name for task in jobs]

        segment_methods = [name for name in method_names if "Segment" in name or "Selector" in name]
        self.assertGreater(len(segment_methods), 0, "Should include segment selector methods")

    def test_get_distributed_jobs_includes_fast_methods(self):
        extraction_data = self.create_sample_extraction_data()

        jobs = self.extractor.get_distributed_jobs(extraction_data)
        method_names = [task.method_name for task in jobs]

        fast_methods = [name for name in method_names if "Fast" in name]
        self.assertGreater(len(fast_methods), 0, "Should include fast methods")

    def test_get_distributed_jobs_includes_t5_methods(self):
        extraction_data = self.create_sample_extraction_data()
        jobs = self.extractor.get_distributed_jobs(extraction_data)
        t5_jobs = [task for task in jobs if "T5" in task.method_name]
        for task in t5_jobs:
            self.assertTrue(task.gpu_needed, "T5 methods should require GPU")

    def test_get_distributed_jobs_includes_gemini_methods(self):
        extraction_data = self.create_sample_extraction_data()

        jobs = self.extractor.get_distributed_jobs(extraction_data)
        method_names = [task.method_name for task in jobs]

        gemini_methods = [name for name in method_names if "Gemini" in name]
        self.assertGreater(len(gemini_methods), 0, "Should include Gemini methods")

    def test_get_distributed_jobs_with_multilingual_content(self):
        extraction_data = self.create_sample_extraction_data()
        extraction_data.samples[0].labeled_data.language_iso = "fr"
        extraction_data.samples[1].labeled_data.language_iso = "de"

        jobs = self.extractor.get_distributed_jobs(extraction_data)

        self.assertIsInstance(jobs, list)
        for task in jobs:
            self.assertEqual(task.extractor_name, "PdfToTextExtractor")

    def test_get_distributed_jobs_method_names_are_unique(self):
        extraction_data = self.create_sample_extraction_data()

        jobs = self.extractor.get_distributed_jobs(extraction_data)

        method_names = [task.method_name for task in jobs]
        unique_method_names = set(method_names)
        self.assertEqual(len(method_names), len(unique_method_names), "All method names should be unique")

    def test_get_distributed_jobs_timeout_values_are_reasonable(self):
        extraction_data = self.create_sample_extraction_data()

        jobs = self.extractor.get_distributed_jobs(extraction_data)

        for task in jobs:
            self.assertGreaterEqual(task.timeout, 300, "Timeout should be at least 5 minutes")
            self.assertLessEqual(task.timeout, 86400, "Timeout should not exceed 24 hours")

    def test_get_distributed_jobs_gpu_distribution(self):
        extraction_data = self.create_sample_extraction_data()

        jobs = self.extractor.get_distributed_jobs(extraction_data)

        gpu_jobs = [task for task in jobs if task.gpu_needed]
        cpu_jobs = [task for task in jobs if not task.gpu_needed]

        self.assertGreater(len(cpu_jobs), 0, "Should have some CPU-only jobs")

    def test_get_distributed_jobs_with_minimal_samples(self):
        extraction_data = self.create_sample_extraction_data(samples_count=1)

        jobs = self.extractor.get_distributed_jobs(extraction_data)

        self.assertIsInstance(jobs, list)

    def test_get_distributed_jobs_with_many_samples(self):
        extraction_data = self.create_sample_extraction_data(samples_count=50)

        jobs = self.extractor.get_distributed_jobs(extraction_data)

        self.assertIsInstance(jobs, list)
        for task in jobs:
            self.assertEqual(task.extractor_name, "PdfToTextExtractor")

    def test_get_distributed_jobs_includes_near_methods(self):
        extraction_data = self.create_sample_extraction_data()

        jobs = self.extractor.get_distributed_jobs(extraction_data)
        method_names = [task.method_name for task in jobs]

        near_methods = [name for name in method_names if "Near" in name or "Position" in name]
        self.assertGreater(len(near_methods), 0, "Should include near/position methods")

    def test_get_distributed_jobs_all_have_valid_extractors(self):
        extraction_data = self.create_sample_extraction_data()

        jobs = self.extractor.get_distributed_jobs(extraction_data)

        for task in jobs:
            self.assertEqual(task.extractor_name, "PdfToTextExtractor")
            self.assertEqual(task.run_name, "default")
            self.assertEqual(task.extraction_name, extraction_id)

    def test_get_distributed_jobs_all_methods_have_valid_properties(self):
        extraction_data = self.create_sample_extraction_data()

        jobs = self.extractor.get_distributed_jobs(extraction_data)

        for job in jobs:
            self.assertIsInstance(job.options, list)
            self.assertIsInstance(job.multi_value, bool)
            self.assertIsInstance(job.should_be_retrained_with_more_data, bool)
