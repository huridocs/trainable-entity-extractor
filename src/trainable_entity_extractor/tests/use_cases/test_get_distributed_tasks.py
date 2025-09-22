import shutil
from unittest import TestCase
from unittest.mock import Mock, patch

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.drivers.TrainableEntityExtractor import TrainableEntityExtractor

extraction_id = "test_get_distributed_jobs"
extraction_identifier = ExtractionIdentifier(extraction_name=extraction_id)


class TestGetDistributedJobs(TestCase):
    def setUp(self):
        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)
        self.trainable_extractor = TrainableEntityExtractor(extraction_identifier)

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

    @patch("trainable_entity_extractor.use_cases.TrainableEntityExtractor.send_logs")
    def test_get_distributed_jobs_with_valid_extractor(self, mock_send_logs):
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

        with patch.object(self.trainable_extractor, "EXTRACTORS", [Mock(return_value=mock_extractor_instance)]):
            jobs = self.trainable_extractor.get_jobs(extraction_data)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].extractor_name, "MockExtractor")
        mock_extractor_instance.can_be_used.assert_called_once_with(extraction_data)
        mock_extractor_instance.get_distributed_jobs.assert_called_once_with(extraction_data)
        mock_send_logs.assert_called_with(extraction_identifier, "Getting jobs for extractor MockExtractor")

    @patch("trainable_entity_extractor.use_cases.TrainableEntityExtractor.send_logs")
    def test_get_distributed_jobs_no_compatible_extractor(self, mock_send_logs):
        extraction_data = self.create_sample_extraction_data()

        mock_extractor_instance = Mock()
        mock_extractor_instance.can_be_used.return_value = False
        mock_extractor_instance.get_name.return_value = "IncompatibleExtractor"

        with patch.object(self.trainable_extractor, "EXTRACTORS", [Mock(return_value=mock_extractor_instance)]):
            jobs = self.trainable_extractor.get_jobs(extraction_data)

        self.assertEqual(len(jobs), 0)
        mock_extractor_instance.can_be_used.assert_called_once_with(extraction_data)
        mock_extractor_instance.get_distributed_jobs.assert_not_called()
        mock_send_logs.assert_not_called()

    @patch("trainable_entity_extractor.use_cases.TrainableEntityExtractor.send_logs")
    def test_get_distributed_jobs_multiple_extractors_first_compatible(self, mock_send_logs):
        extraction_data = self.create_sample_extraction_data()

        mock_extractor1 = Mock()
        mock_extractor1.can_be_used.return_value = True
        mock_extractor1.get_name.return_value = "FirstExtractor"
        mock_task = TrainableEntityExtractorJob(
            run_name="test_run",
            extraction_name="test_extraction",
            extractor_name="FirstExtractor",
            method_name="test_method",
            gpu_needed=False,
            timeout=300,
        )
        mock_extractor1.get_distributed_jobs.return_value = [mock_task]

        mock_extractor2 = Mock()
        mock_extractor2.can_be_used.return_value = True
        mock_extractor2.get_name.return_value = "SecondExtractor"

        with patch.object(
            self.trainable_extractor, "EXTRACTORS", [Mock(return_value=mock_extractor1), Mock(return_value=mock_extractor2)]
        ):
            jobs = self.trainable_extractor.get_jobs(extraction_data)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].extractor_name, "FirstExtractor")
        mock_extractor1.can_be_used.assert_called_once_with(extraction_data)
        mock_extractor1.get_distributed_jobs.assert_called_once_with(extraction_data)
        mock_extractor2.can_be_used.assert_not_called()
        mock_extractor2.get_distributed_jobs.assert_not_called()

    @patch("trainable_entity_extractor.use_cases.TrainableEntityExtractor.send_logs")
    def test_get_distributed_jobs_multiple_extractors_second_compatible(self, mock_send_logs):
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

        with patch.object(
            self.trainable_extractor, "EXTRACTORS", [Mock(return_value=mock_extractor1), Mock(return_value=mock_extractor2)]
        ):
            jobs = self.trainable_extractor.get_jobs(extraction_data)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].extractor_name, "SecondExtractor")
        self.assertTrue(jobs[0].gpu_needed)
        self.assertEqual(jobs[0].timeout, 600)
        mock_extractor1.can_be_used.assert_called_once_with(extraction_data)
        mock_extractor2.can_be_used.assert_called_once_with(extraction_data)
        mock_extractor2.get_distributed_jobs.assert_called_once_with(extraction_data)

    @patch("trainable_entity_extractor.use_cases.TrainableEntityExtractor.send_logs")
    def test_get_distributed_jobs_empty_extraction_data(self, mock_send_logs):
        extraction_data = ExtractionData(
            samples=[], options=[], multi_value=False, extraction_identifier=extraction_identifier
        )

        mock_extractor_instance = Mock()
        mock_extractor_instance.can_be_used.return_value = False

        with patch.object(self.trainable_extractor, "EXTRACTORS", [Mock(return_value=mock_extractor_instance)]):
            jobs = self.trainable_extractor.get_jobs(extraction_data)

        self.assertEqual(len(jobs), 0)
        mock_extractor_instance.can_be_used.assert_called_once_with(extraction_data)

    @patch("trainable_entity_extractor.use_cases.TrainableEntityExtractor.send_logs")
    def test_get_distributed_jobs_extractor_returns_multiple_jobs(self, mock_send_logs):
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

        with patch.object(self.trainable_extractor, "EXTRACTORS", [Mock(return_value=mock_extractor_instance)]):
            jobs = self.trainable_extractor.get_jobs(extraction_data)

        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0].method_name, "method_1")
        self.assertEqual(jobs[1].method_name, "method_2")
        self.assertFalse(jobs[0].gpu_needed)
        self.assertTrue(jobs[1].gpu_needed)
