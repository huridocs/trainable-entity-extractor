import shutil
from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionDistributedTask import ExtractionDistributedTask
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.extractors.pdf_to_text_extractor.PdfToTextExtractor import PdfToTextExtractor

extraction_id = "test_pdf_to_text"
extraction_identifier = ExtractionIdentifier(extraction_name=extraction_id)


class TestPdfToTextExtractorDistributedTasks(TestCase):
    def setUp(self):
        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)
        self.extractor = PdfToTextExtractor(extraction_identifier)

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

    def test_get_distributed_tasks_returns_tasks(self):
        extraction_data = self.create_sample_extraction_data()

        tasks = self.extractor.get_distributed_tasks(extraction_data)

        self.assertIsInstance(tasks, list)
        self.assertGreater(len(tasks), 0, "Should return at least one task")

    def test_get_distributed_tasks_creates_valid_task_objects(self):
        extraction_data = self.create_sample_extraction_data()

        tasks = self.extractor.get_distributed_tasks(extraction_data)

        for task in tasks:
            self.assertIsInstance(task, ExtractionDistributedTask)
            self.assertEqual(task.run_name, extraction_data.extraction_identifier.run_name)
            self.assertEqual(task.extraction_name, extraction_data.extraction_identifier.extraction_name)
            self.assertEqual(task.extractor_name, "PdfToTextExtractor")
            self.assertIsInstance(task.method_name, str)
            self.assertGreater(len(task.method_name), 0, "Method name should not be empty")
            self.assertIsInstance(task.gpu_needed, bool)
            self.assertIsInstance(task.timeout, int)
            self.assertGreater(task.timeout, 0, "Timeout should be positive")

    def test_get_distributed_tasks_includes_segment_selector_methods(self):
        extraction_data = self.create_sample_extraction_data()

        tasks = self.extractor.get_distributed_tasks(extraction_data)
        method_names = [task.method_name for task in tasks]

        segment_methods = [name for name in method_names if "Segment" in name or "Selector" in name]
        self.assertGreater(len(segment_methods), 0, "Should include segment selector methods")

    def test_get_distributed_tasks_includes_fast_methods(self):
        extraction_data = self.create_sample_extraction_data()

        tasks = self.extractor.get_distributed_tasks(extraction_data)
        method_names = [task.method_name for task in tasks]

        fast_methods = [name for name in method_names if "Fast" in name]
        self.assertGreater(len(fast_methods), 0, "Should include fast methods")

    def test_get_distributed_tasks_includes_t5_methods(self):
        extraction_data = self.create_sample_extraction_data()
        tasks = self.extractor.get_distributed_tasks(extraction_data)
        t5_tasks = [task for task in tasks if "T5" in task.method_name]
        for task in t5_tasks:
            self.assertTrue(task.gpu_needed, "T5 methods should require GPU")

    def test_get_distributed_tasks_includes_gemini_methods(self):
        extraction_data = self.create_sample_extraction_data()

        tasks = self.extractor.get_distributed_tasks(extraction_data)
        method_names = [task.method_name for task in tasks]

        gemini_methods = [name for name in method_names if "Gemini" in name]
        self.assertGreater(len(gemini_methods), 0, "Should include Gemini methods")

    def test_get_distributed_tasks_with_multilingual_content(self):
        extraction_data = self.create_sample_extraction_data()
        extraction_data.samples[0].labeled_data.language_iso = "fr"
        extraction_data.samples[1].labeled_data.language_iso = "de"

        tasks = self.extractor.get_distributed_tasks(extraction_data)

        self.assertIsInstance(tasks, list)
        for task in tasks:
            self.assertEqual(task.extractor_name, "PdfToTextExtractor")

    def test_get_distributed_tasks_method_names_are_unique(self):
        extraction_data = self.create_sample_extraction_data()

        tasks = self.extractor.get_distributed_tasks(extraction_data)

        method_names = [task.method_name for task in tasks]
        unique_method_names = set(method_names)
        self.assertEqual(len(method_names), len(unique_method_names), "All method names should be unique")

    def test_get_distributed_tasks_timeout_values_are_reasonable(self):
        extraction_data = self.create_sample_extraction_data()

        tasks = self.extractor.get_distributed_tasks(extraction_data)

        for task in tasks:
            self.assertGreaterEqual(task.timeout, 300, "Timeout should be at least 5 minutes")
            self.assertLessEqual(task.timeout, 86400, "Timeout should not exceed 24 hours")

    def test_get_distributed_tasks_gpu_distribution(self):
        extraction_data = self.create_sample_extraction_data()

        tasks = self.extractor.get_distributed_tasks(extraction_data)

        gpu_tasks = [task for task in tasks if task.gpu_needed]
        cpu_tasks = [task for task in tasks if not task.gpu_needed]

        self.assertGreater(len(cpu_tasks), 0, "Should have some CPU-only tasks")

    def test_get_distributed_tasks_with_minimal_samples(self):
        extraction_data = self.create_sample_extraction_data(samples_count=1)

        tasks = self.extractor.get_distributed_tasks(extraction_data)

        self.assertIsInstance(tasks, list)

    def test_get_distributed_tasks_with_many_samples(self):
        extraction_data = self.create_sample_extraction_data(samples_count=50)

        tasks = self.extractor.get_distributed_tasks(extraction_data)

        self.assertIsInstance(tasks, list)
        for task in tasks:
            self.assertEqual(task.extractor_name, "PdfToTextExtractor")

    def test_get_distributed_tasks_includes_near_methods(self):
        extraction_data = self.create_sample_extraction_data()

        tasks = self.extractor.get_distributed_tasks(extraction_data)
        method_names = [task.method_name for task in tasks]

        near_methods = [name for name in method_names if "Near" in name or "Position" in name]
        self.assertGreater(len(near_methods), 0, "Should include near/position methods")

    def test_get_distributed_tasks_all_have_valid_extractors(self):
        extraction_data = self.create_sample_extraction_data()

        tasks = self.extractor.get_distributed_tasks(extraction_data)

        for task in tasks:
            self.assertEqual(task.extractor_name, "PdfToTextExtractor")
            self.assertEqual(task.run_name, "default")
            self.assertEqual(task.extraction_name, extraction_id)
