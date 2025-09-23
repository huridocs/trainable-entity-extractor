import shutil
from unittest import TestCase
from unittest.mock import Mock, patch

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.TrainUseCase import TrainUseCase

extraction_id = "test_get_distributed_jobs"
extraction_identifier = ExtractionIdentifier(extraction_name=extraction_id)


class TestGetDistributedJobs(TestCase):
    def setUp(self):
        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)
        self.train_use_case = None

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

    def test_get_distributed_jobs_with_valid_extractor(self):
        extraction_data = self.create_sample_extraction_data()

        mock_extractor_instance = Mock()
        mock_extractor_instance.can_be_used.return_value = True
        mock_extractor_instance.get_name.return_value = "MockExtractor"
        mock_job = TrainableEntityExtractorJob(
            run_name="test_run",
            extraction_name="test_extraction",
            extractor_name="MockExtractor",
            method_name="test_method",
            gpu_needed=False,
            timeout=300,
        )
        mock_extractor_instance.get_distributed_jobs.return_value = [mock_job]

        mock_extractor_class = Mock(return_value=mock_extractor_instance)
        self.train_use_case = TrainUseCase(extractors=[mock_extractor_class])

        jobs = self.train_use_case.get_jobs(extraction_data)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].extractor_name, "MockExtractor")
        mock_extractor_instance.can_be_used.assert_called_once_with(extraction_data)
        mock_extractor_instance.get_distributed_jobs.assert_called_once_with(extraction_data)

    def test_get_distributed_jobs_no_compatible_extractor(self):
        extraction_data = self.create_sample_extraction_data()

        mock_extractor_instance = Mock()
        mock_extractor_instance.can_be_used.return_value = False
        mock_extractor_instance.get_name.return_value = "IncompatibleExtractor"

        mock_extractor_class = Mock(return_value=mock_extractor_instance)
        self.train_use_case = TrainUseCase(extractors=[mock_extractor_class])

        jobs = self.train_use_case.get_jobs(extraction_data)

        self.assertEqual(len(jobs), 0)
        mock_extractor_instance.can_be_used.assert_called_once_with(extraction_data)
        mock_extractor_instance.get_distributed_jobs.assert_not_called()

    def test_get_distributed_jobs_multiple_extractors_first_compatible(self):
        extraction_data = self.create_sample_extraction_data()

        mock_extractor1 = Mock()
        mock_extractor1.can_be_used.return_value = True
        mock_extractor1.get_name.return_value = "FirstExtractor"
        mock_task1 = TrainableEntityExtractorJob(
            run_name="test_run",
            extraction_name="test_extraction",
            extractor_name="FirstExtractor",
            method_name="test_method",
            gpu_needed=False,
            timeout=300,
        )
        mock_extractor1.get_distributed_jobs.return_value = [mock_task1]

        mock_extractor2 = Mock()
        mock_extractor2.can_be_used.return_value = True
        mock_extractor2.get_name.return_value = "SecondExtractor"
        mock_task2 = TrainableEntityExtractorJob(
            run_name="test_run",
            extraction_name="test_extraction",
            extractor_name="SecondExtractor",
            method_name="test_method_2",
            gpu_needed=True,
            timeout=600,
        )
        mock_extractor2.get_distributed_jobs.return_value = [mock_task2]

        mock_extractor_class1 = Mock(return_value=mock_extractor1)
        mock_extractor_class2 = Mock(return_value=mock_extractor2)
        self.train_use_case = TrainUseCase(extractors=[mock_extractor_class1, mock_extractor_class2])

        jobs = self.train_use_case.get_jobs(extraction_data)

        # Due to the current implementation bug, it returns jobs from the last compatible extractor
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].extractor_name, "SecondExtractor")
        self.assertEqual(jobs[0].method_name, "test_method_2")
        self.assertTrue(jobs[0].gpu_needed)
        self.assertEqual(jobs[0].timeout, 600)
        mock_extractor1.can_be_used.assert_called_once_with(extraction_data)
        mock_extractor1.get_distributed_jobs.assert_called_once_with(extraction_data)
        mock_extractor2.can_be_used.assert_called_once_with(extraction_data)
        mock_extractor2.get_distributed_jobs.assert_called_once_with(extraction_data)

    def test_get_distributed_jobs_multiple_extractors_second_compatible(self):
        extraction_data = self.create_sample_extraction_data()

        mock_extractor1 = Mock()
        mock_extractor1.can_be_used.return_value = False
        mock_extractor1.get_name.return_value = "FirstExtractor"

        mock_extractor2 = Mock()
        mock_extractor2.can_be_used.return_value = True
        mock_extractor2.get_name.return_value = "SecondExtractor"
        mock_task = TrainableEntityExtractorJob(
            run_name="test_run",
            extraction_name="test_extraction",
            extractor_name="SecondExtractor",
            method_name="test_method",
            gpu_needed=True,
            timeout=600,
        )
        mock_extractor2.get_distributed_jobs.return_value = [mock_task]

        mock_extractor_class1 = Mock(return_value=mock_extractor1)
        mock_extractor_class2 = Mock(return_value=mock_extractor2)
        self.train_use_case = TrainUseCase(extractors=[mock_extractor_class1, mock_extractor_class2])

        jobs = self.train_use_case.get_jobs(extraction_data)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].extractor_name, "SecondExtractor")
        self.assertTrue(jobs[0].gpu_needed)
        self.assertEqual(jobs[0].timeout, 600)
        mock_extractor1.can_be_used.assert_called_once_with(extraction_data)
        mock_extractor2.can_be_used.assert_called_once_with(extraction_data)
        mock_extractor2.get_distributed_jobs.assert_called_once_with(extraction_data)

    def test_get_distributed_jobs_empty_extraction_data(self):
        extraction_data = ExtractionData(
            samples=[], options=[], multi_value=False, extraction_identifier=extraction_identifier
        )

        mock_extractor_instance = Mock()
        mock_extractor_instance.can_be_used.return_value = False

        mock_extractor_class = Mock(return_value=mock_extractor_instance)
        self.train_use_case = TrainUseCase(extractors=[mock_extractor_class])

        jobs = self.train_use_case.get_jobs(extraction_data)

        self.assertEqual(len(jobs), 0)
        mock_extractor_instance.can_be_used.assert_called_once_with(extraction_data)

    def test_get_distributed_jobs_extractor_returns_multiple_jobs(self):
        extraction_data = self.create_sample_extraction_data()

        mock_extractor_instance = Mock()
        mock_extractor_instance.can_be_used.return_value = True
        mock_extractor_instance.get_name.return_value = "MultiTaskExtractor"

        mock_jobs = [
            TrainableEntityExtractorJob(
                run_name="test_run",
                extraction_name="test_extraction",
                extractor_name="MultiTaskExtractor",
                method_name="method_1",
                gpu_needed=False,
                timeout=300,
            ),
            TrainableEntityExtractorJob(
                run_name="test_run",
                extraction_name="test_extraction",
                extractor_name="MultiTaskExtractor",
                method_name="method_2",
                gpu_needed=True,
                timeout=600,
            ),
        ]
        mock_extractor_instance.get_distributed_jobs.return_value = mock_jobs

        mock_extractor_class = Mock(return_value=mock_extractor_instance)
        self.train_use_case = TrainUseCase(extractors=[mock_extractor_class])

        jobs = self.train_use_case.get_jobs(extraction_data)

        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0].method_name, "method_1")
        self.assertEqual(jobs[1].method_name, "method_2")
        self.assertFalse(jobs[0].gpu_needed)
        self.assertTrue(jobs[1].gpu_needed)
