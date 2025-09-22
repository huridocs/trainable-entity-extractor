import shutil
from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.PdfToMultiOptionExtractor import (
    PdfToMultiOptionExtractor,
)

extraction_id = "test_pdf_to_multi_option"
extraction_identifier = ExtractionIdentifier(extraction_name=extraction_id)


class TestPdfToMultiOptionExtractorDistributedJobs(TestCase):
    def setUp(self):
        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)
        self.extractor = PdfToMultiOptionExtractor(extraction_identifier)

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
                label_text=f"label_{i}",
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
            self.assertEqual(task.extractor_name, "PdfToMultiOptionExtractor")
            self.assertIsInstance(task.method_name, str)
            self.assertGreater(len(task.method_name), 0, "Method name should not be empty")
            self.assertIsInstance(task.gpu_needed, bool)
            self.assertIsInstance(task.timeout, int)
            self.assertGreater(task.timeout, 0, "Timeout should be positive")

    def test_get_distributed_jobs_with_multilingual_data(self):
        extraction_data = self.create_sample_extraction_data()
        extraction_data.samples[0].labeled_data.language_iso = "fr"
        extraction_data.samples[1].labeled_data.language_iso = "de"

        jobs = self.extractor.get_distributed_jobs(extraction_data)

        self.assertIsInstance(jobs, list)
        self.assertGreater(len(jobs), 0)
        for task in jobs:
            self.assertEqual(task.extractor_name, "PdfToMultiOptionExtractor")

    def test_get_distributed_jobs_with_multi_value_data(self):
        extraction_data = self.create_sample_extraction_data()
        extraction_data.multi_value = True

        jobs = self.extractor.get_distributed_jobs(extraction_data)

        self.assertIsInstance(jobs, list)
        for task in jobs:
            self.assertEqual(task.extractor_name, "PdfToMultiOptionExtractor")

    def test_get_distributed_jobs_with_empty_options(self):
        extraction_data = self.create_sample_extraction_data()
        extraction_data.options = []

        jobs = self.extractor.get_distributed_jobs(extraction_data)

        self.assertIsInstance(jobs, list)

    def test_get_distributed_jobs_with_minimal_samples(self):
        extraction_data = self.create_sample_extraction_data(samples_count=1)

        jobs = self.extractor.get_distributed_jobs(extraction_data)

        self.assertIsInstance(jobs, list)

    def test_get_distributed_jobs_with_many_samples(self):
        extraction_data = self.create_sample_extraction_data(samples_count=20)

        jobs = self.extractor.get_distributed_jobs(extraction_data)

        self.assertIsInstance(jobs, list)
        for task in jobs:
            self.assertEqual(task.extractor_name, "PdfToMultiOptionExtractor")

    def test_get_distributed_jobs_method_names_are_unique(self):
        extraction_data = self.create_sample_extraction_data()

        jobs = self.extractor.get_distributed_jobs(extraction_data)

        method_names = [task.method_name for task in jobs]
        unique_method_names = set(method_names)
        self.assertEqual(len(method_names), len(unique_method_names), "All method names should be unique")

    def test_get_distributed_jobs_includes_expected_method_types(self):
        extraction_data = self.create_sample_extraction_data()

        jobs = self.extractor.get_distributed_jobs(extraction_data)
        method_names = [task.method_name for task in jobs]

        fuzzy_methods = [name for name in method_names if "Fuzzy" in name]
        self.assertGreater(len(fuzzy_methods), 0, "Should include fuzzy methods")

    def test_get_distributed_jobs_timeout_values_are_reasonable(self):
        extraction_data = self.create_sample_extraction_data()

        jobs = self.extractor.get_distributed_jobs(extraction_data)

        for task in jobs:
            self.assertGreaterEqual(task.timeout, 300, "Timeout should be at least 5 minutes")
            self.assertLessEqual(task.timeout, 86400, "Timeout should not exceed 24 hours")

    def test_get_distributed_jobs_with_different_languages(self):
        extraction_data = self.create_sample_extraction_data()
        languages = ["en", "fr", "de", "es", "it"]
        for i, sample in enumerate(extraction_data.samples):
            sample.labeled_data.language_iso = languages[i % len(languages)]

        jobs = self.extractor.get_distributed_jobs(extraction_data)

        self.assertIsInstance(jobs, list)
        for task in jobs:
            self.assertEqual(task.extractor_name, "PdfToMultiOptionExtractor")
