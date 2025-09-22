import shutil
from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.TextToTextExtractor import TextToTextExtractor

extraction_id = "test_text_to_text"
extraction_identifier = ExtractionIdentifier(extraction_name=extraction_id)


class TestTextToTextExtractorDistributedJobs(TestCase):
    def setUp(self):
        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)
        self.extractor = TextToTextExtractor(extraction_identifier)

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
                source_text=f"This is source text for extraction {i}. Date: 2023-{i+1:02d}-15",
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
            self.assertEqual(task.extractor_name, "TextToTextExtractor")
            self.assertIsInstance(task.method_name, str)
            self.assertGreater(len(task.method_name), 0, "Method name should not be empty")
            self.assertIsInstance(task.gpu_needed, bool)
            self.assertIsInstance(task.timeout, int)
            self.assertGreater(task.timeout, 0, "Timeout should be positive")

    def test_get_distributed_jobs_includes_regex_methods(self):
        extraction_data = self.create_sample_extraction_data()

        jobs = self.extractor.get_distributed_jobs(extraction_data)
        method_names = [task.method_name for task in jobs]

        regex_methods = [name for name in method_names if "Regex" in name]
        self.assertGreater(len(regex_methods), 0, "Should include regex methods")

    def test_get_distributed_jobs_includes_date_parser_methods(self):
        extraction_data = self.create_sample_extraction_data()

        jobs = self.extractor.get_distributed_jobs(extraction_data)
        method_names = [task.method_name for task in jobs]

        date_methods = [name for name in method_names if "Date" in name or "Parser" in name]
        self.assertGreater(len(date_methods), 0, "Should include date parser methods")

    def test_get_distributed_jobs_includes_ner_methods(self):
        extraction_data = self.create_sample_extraction_data()

        jobs = self.extractor.get_distributed_jobs(extraction_data)
        method_names = [task.method_name for task in jobs]

        ner_methods = [name for name in method_names if "Ner" in name]
        self.assertGreater(len(ner_methods), 0, "Should include NER methods")

    def test_get_distributed_jobs_includes_same_input_output_method(self):
        extraction_data = self.create_sample_extraction_data()

        jobs = self.extractor.get_distributed_jobs(extraction_data)
        method_names = [task.method_name for task in jobs]

        same_io_methods = [name for name in method_names if "SameInput" in name or "Input" in name]
        self.assertGreater(len(same_io_methods), 0, "Should include same input/output methods")

    def test_get_distributed_jobs_with_date_content(self):
        extraction_data = self.create_sample_extraction_data()
        for sample in extraction_data.samples:
            sample.labeled_data.source_text = "Document date: January 15, 2023"
            sample.labeled_data.label_text = "2023-01-15"

        jobs = self.extractor.get_distributed_jobs(extraction_data)

        self.assertIsInstance(jobs, list)
        self.assertGreater(len(jobs), 0)

    def test_get_distributed_jobs_with_multilingual_content(self):
        extraction_data = self.create_sample_extraction_data()
        extraction_data.samples[0].labeled_data.language_iso = "fr"
        extraction_data.samples[1].labeled_data.language_iso = "de"

        jobs = self.extractor.get_distributed_jobs(extraction_data)

        self.assertIsInstance(jobs, list)
        for task in jobs:
            self.assertEqual(task.extractor_name, "TextToTextExtractor")

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

    def test_get_distributed_jobs_gpu_requirements_are_boolean(self):
        extraction_data = self.create_sample_extraction_data()

        jobs = self.extractor.get_distributed_jobs(extraction_data)

        for task in jobs:
            self.assertIsInstance(task.gpu_needed, bool, "GPU requirement should be boolean")

    def test_get_distributed_jobs_with_gliner_methods(self):
        extraction_data = self.create_sample_extraction_data()

        jobs = self.extractor.get_distributed_jobs(extraction_data)
        method_names = [task.method_name for task in jobs]

        gliner_methods = [name for name in method_names if "Gliner" in name]
        if len(gliner_methods) > 0:
            gliner_jobs = [task for task in jobs if "Gliner" in task.method_name]
            for task in gliner_jobs:
                self.assertTrue(task.gpu_needed, "Gliner methods should require GPU")

    def test_get_distributed_jobs_with_empty_label_text(self):
        extraction_data = self.create_sample_extraction_data()
        for sample in extraction_data.samples:
            sample.labeled_data.label_text = ""

        jobs = self.extractor.get_distributed_jobs(extraction_data)

        self.assertIsInstance(jobs, list)
