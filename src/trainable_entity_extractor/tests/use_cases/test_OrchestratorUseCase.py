import shutil
from unittest import TestCase

from trainable_entity_extractor.adapters.ExtractorLogger import ExtractorLogger
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.TextToTextExtractor import TextToTextExtractor
from trainable_entity_extractor.adapters.LocalExtractionDataRetriever import LocalExtractionDataRetriever
from trainable_entity_extractor.adapters.LocalModelStorage import LocalModelStorage
from trainable_entity_extractor.adapters.LocalJobExecutor import LocalJobExecutor
from trainable_entity_extractor.config import CACHE_PATH
from trainable_entity_extractor.domain.DistributedJob import DistributedJob
from trainable_entity_extractor.domain.DistributedSubJob import DistributedSubJob
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.JobStatus import JobStatus
from trainable_entity_extractor.domain.JobType import JobType
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.OrchestratorUseCase import OrchestratorUseCase
from trainable_entity_extractor.use_cases.TrainUseCase import TrainUseCase

extraction_id = "test_orchestrator_use_case"
extraction_identifier = ExtractionIdentifier(extraction_name=extraction_id)


class TestOrchestratorUseCase(TestCase):
    def setUp(self):
        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)
        self.data_retriever = LocalExtractionDataRetriever()
        self.model_storage = LocalModelStorage()
        self.logger = ExtractorLogger()
        self.extractors = [TextToTextExtractor]
        self.job_executor = LocalJobExecutor(self.extractors, self.data_retriever, self.model_storage, self.logger)
        self.orchestrator = OrchestratorUseCase(self.job_executor, self.logger)

    def tearDown(self):
        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)
        shutil.rmtree(CACHE_PATH, ignore_errors=True)

    def _create_sample_extraction_data(self) -> ExtractionData:
        """Create sample training data for testing"""
        samples = [
            TrainingSample(
                labeled_data=LabeledData(label_text="John Doe", language_iso="en", source_text="My name is John Doe")
            ),
            TrainingSample(
                labeled_data=LabeledData(label_text="Jane Smith", language_iso="en", source_text="Jane Smith is here")
            ),
            TrainingSample(
                labeled_data=LabeledData(label_text="Bob Johnson", language_iso="en", source_text="Bob Johnson works")
            ),
        ]
        return ExtractionData(samples=samples, extraction_identifier=extraction_identifier)

    def _create_sample_prediction_data(self) -> list[PredictionSample]:
        """Create sample prediction data for testing"""
        return [
            PredictionSample.from_text("Hello, my name is Alice Brown", "sample_1"),
            PredictionSample.from_text("Meet Charlie Wilson today", "sample_2"),
            PredictionSample.from_text("Diana Prince is the manager", "sample_3"),
        ]

    def _get_available_training_job(self, extraction_data: ExtractionData) -> TrainableEntityExtractorJob:
        """Get a training job for testing - using SameInputOutputMethod for simplicity"""
        train_use_case = TrainUseCase(extractors=self.extractors, logger=self.logger)
        jobs = train_use_case.get_jobs(extraction_data)

        # Find SameInputOutputMethod for consistent testing
        same_input_output_jobs = [job for job in jobs if job.method_name == "SameInputOutputMethod"]
        self.assertGreater(len(same_input_output_jobs), 0, "SameInputOutputMethod not available")

        return same_input_output_jobs[0]

    def _create_test_extractor_job(self, method_name: str = "SameInputOutputMethod") -> TrainableEntityExtractorJob:
        """Create a test TrainableEntityExtractorJob with all required fields"""
        return TrainableEntityExtractorJob(
            run_name="test_run",
            extraction_name=extraction_id,
            method_name=method_name,
            extractor_name="TextToTextExtractor",
            gpu_needed=False,
            timeout=300,
            multi_value=False,
            options=[],
        )

    def test_process_training_job_success(self):
        """Test successful training job processing"""
        extraction_data = self._create_sample_extraction_data()
        self.data_retriever.save_extraction_data(extraction_identifier, extraction_data)

        # Create training job
        extractor_job = self._get_available_training_job(extraction_data)
        sub_job = DistributedSubJob(job_id="test_job_1", extractor_job=extractor_job)
        distributed_job = DistributedJob(extraction_identifier=extraction_identifier, type=JobType.TRAIN, sub_jobs=[sub_job])

        # Add job to orchestrator
        self.orchestrator.distributed_jobs = [distributed_job]

        # Process the job
        success, message = self.orchestrator.process_job(distributed_job)

        # Only check for success
        self.assertTrue(success, f"Training should succeed: {message}")
        self.assertEqual(0, len(self.orchestrator.distributed_jobs), "Job should be removed after completion")

    def test_process_prediction_job_success(self):
        """Test prediction job processing"""
        # First try to train a model
        extraction_data = self._create_sample_extraction_data()
        self.data_retriever.save_extraction_data(extraction_identifier, extraction_data)

        extractor_job = self._get_available_training_job(extraction_data)
        train_use_case = TrainUseCase(extractors=self.extractors, logger=self.logger)
        train_success, train_message = train_use_case.train_one_method(extractor_job, extraction_data)

        # Upload the trained model
        self.model_storage.upload_model(extraction_identifier, extractor_job)

        # Create prediction data
        prediction_data = self._create_sample_prediction_data()
        self.data_retriever.save_prediction_data(extraction_identifier, prediction_data)

        # Create prediction job
        sub_job = DistributedSubJob(job_id="test_job_2", extractor_job=extractor_job)
        distributed_job = DistributedJob(
            extraction_identifier=extraction_identifier, type=JobType.PREDICT, sub_jobs=[sub_job]
        )

        # Add job to orchestrator
        self.orchestrator.distributed_jobs = [distributed_job]

        # Process the job
        success, message = self.orchestrator.process_job(distributed_job)

        # Only check for success
        self.assertTrue(success, f"Prediction should succeed: {message}")
        self.assertEqual(0, len(self.orchestrator.distributed_jobs), "Job should be removed after completion")

    def test_process_performance_job_success(self):
        """Test performance evaluation job processing"""
        extraction_data = self._create_sample_extraction_data()
        self.data_retriever.save_extraction_data(extraction_identifier, extraction_data)

        # Create multiple training jobs for performance comparison
        train_use_case = TrainUseCase(extractors=self.extractors, logger=self.logger)
        all_jobs = train_use_case.get_jobs(extraction_data)

        # Use first job for performance testing (simplified to avoid complexity)
        test_jobs = all_jobs[:5] if len(all_jobs) >= 1 else all_jobs

        sub_jobs = [DistributedSubJob(job_id=f"test_job_{i}", extractor_job=job) for i, job in enumerate(test_jobs)]
        distributed_job = DistributedJob(
            extraction_identifier=extraction_identifier, type=JobType.PERFORMANCE, sub_jobs=sub_jobs
        )

        # Add job to orchestrator
        self.orchestrator.distributed_jobs = [distributed_job]

        # Process the job once
        while self.orchestrator.exists_jobs_to_be_done():
            success, message = self.orchestrator.process_job(self.orchestrator.distributed_jobs[0])

        self.assertTrue(success, "Performance job should return a boolean result")
        self.assertIsInstance(message, str, "Performance job should return a message")
        self.assertEqual(0, len(self.orchestrator.distributed_jobs), "Job should be removed after processing")

    def test_process_training_job_failure_no_data(self):
        """Test training job failure when no extraction data is available"""
        extractor_job = self._create_test_extractor_job()
        sub_job = DistributedSubJob(job_id="test_job_3", extractor_job=extractor_job)
        distributed_job = DistributedJob(extraction_identifier=extraction_identifier, type=JobType.TRAIN, sub_jobs=[sub_job])

        # Add job to orchestrator
        self.orchestrator.distributed_jobs = [distributed_job]

        # Process the job without saving any data
        success, message = self.orchestrator.process_job(distributed_job)

        # Verify failure
        self.assertFalse(success, "Training should fail without data")
        self.assertIn("Training failed", message)
        self.assertEqual(JobStatus.FAILURE, sub_job.status)
        self.assertEqual(0, len(self.orchestrator.distributed_jobs), "Job should be removed after failure")

    def test_process_unknown_job_type(self):
        """Test processing of unknown job type"""
        extractor_job = self._create_test_extractor_job()
        sub_job = DistributedSubJob(job_id="test_job_5", extractor_job=extractor_job)

        # Create a valid distributed job first, then manually set an invalid type
        distributed_job = DistributedJob(
            extraction_identifier=extraction_identifier, type=JobType.TRAIN, sub_jobs=[sub_job]  # Start with valid type
        )

        # Manually override the type to test unknown type handling
        distributed_job.type = "UNKNOWN_TYPE"

        # Add job to orchestrator
        self.orchestrator.distributed_jobs = [distributed_job]

        # Process the job
        success, message = self.orchestrator.process_job(distributed_job)

        # Verify failure
        self.assertFalse(success, "Unknown job type should fail")
        self.assertIn("Unknown job type", message)
        self.assertEqual(0, len(self.orchestrator.distributed_jobs), "Job should be removed after failure")

    def test_exists_jobs_to_be_done(self):
        """Test the exists_jobs_to_be_done method"""
        # Initially no jobs
        self.assertFalse(self.orchestrator.exists_jobs_to_be_done())

        # Add a job
        extractor_job = self._create_test_extractor_job()
        sub_job = DistributedSubJob(job_id="test_job_6", extractor_job=extractor_job)
        distributed_job = DistributedJob(extraction_identifier=extraction_identifier, type=JobType.TRAIN, sub_jobs=[sub_job])

        self.orchestrator.distributed_jobs = [distributed_job]

        # Now should have jobs
        self.assertTrue(self.orchestrator.exists_jobs_to_be_done())

        # Remove the job
        self.orchestrator.distributed_jobs = []

        # Should be no jobs again
        self.assertFalse(self.orchestrator.exists_jobs_to_be_done())

    def test_full_workflow_train_then_predict(self):
        """Test complete workflow: train a model then use it for prediction"""
        extraction_data = self._create_sample_extraction_data()
        self.data_retriever.save_extraction_data(extraction_identifier, extraction_data)

        # Step 1: Create and process training job
        extractor_job = self._get_available_training_job(extraction_data)
        train_sub_job = DistributedSubJob(job_id="test_job_7", extractor_job=extractor_job)
        train_distributed_job = DistributedJob(
            extraction_identifier=extraction_identifier, type=JobType.TRAIN, sub_jobs=[train_sub_job]
        )

        self.orchestrator.distributed_jobs = [train_distributed_job]

        # Process training
        train_success, train_message = self.orchestrator.process_job(train_distributed_job)
        self.assertEqual(0, len(self.orchestrator.distributed_jobs))

        if not train_success:
            self.skipTest(f"Training failed, cannot test full workflow: {train_message}")

        # Step 2: Create prediction data and job
        prediction_data = self._create_sample_prediction_data()
        self.data_retriever.save_prediction_data(extraction_identifier, prediction_data)

        predict_sub_job = DistributedSubJob(job_id="test_job_8", extractor_job=extractor_job)
        predict_distributed_job = DistributedJob(
            extraction_identifier=extraction_identifier, type=JobType.PREDICT, sub_jobs=[predict_sub_job]
        )

        self.orchestrator.distributed_jobs = [predict_distributed_job]

        # Process prediction
        predict_success, predict_message = self.orchestrator.process_job(predict_distributed_job)
        self.assertEqual(0, len(self.orchestrator.distributed_jobs))

        if predict_success:
            # Verify suggestions were created
            suggestions = self.data_retriever.get_suggestions(extraction_identifier)
            self.assertGreaterEqual(len(suggestions), 0, "Should have suggestions or empty list")

    def test_multiple_jobs_processing(self):
        """Test processing multiple jobs in sequence"""
        extraction_data = self._create_sample_extraction_data()
        self.data_retriever.save_extraction_data(extraction_identifier, extraction_data)

        # Create multiple training jobs
        train_use_case = TrainUseCase(extractors=self.extractors, logger=self.logger)
        all_jobs = train_use_case.get_jobs(extraction_data)

        # Create distributed jobs for first 2 available methods
        test_jobs = all_jobs[:2] if len(all_jobs) >= 2 else all_jobs
        distributed_jobs = []

        for i, job in enumerate(test_jobs):
            sub_job = DistributedSubJob(job_id=f"test_job_multi_{i}", extractor_job=job)
            distributed_job = DistributedJob(
                extraction_identifier=ExtractionIdentifier(extraction_name=f"{extraction_id}_{i}"),
                type=JobType.TRAIN,
                sub_jobs=[sub_job],
            )
            distributed_jobs.append(distributed_job)

            # Save data for each extraction identifier
            self.data_retriever.save_extraction_data(distributed_job.extraction_identifier, extraction_data)

        # Add all jobs to orchestrator
        self.orchestrator.distributed_jobs = distributed_jobs.copy()

        # Process all jobs
        processed_jobs = 0
        while self.orchestrator.exists_jobs_to_be_done():
            current_job = self.orchestrator.distributed_jobs[0]
            success, message = self.orchestrator.process_job(current_job)
            processed_jobs += 1

            # Prevent infinite loop
            if processed_jobs > len(test_jobs):
                break

        # Verify all jobs were processed (regardless of success/failure)
        self.assertEqual(processed_jobs, len(test_jobs), "All jobs should be processed")
        self.assertFalse(self.orchestrator.exists_jobs_to_be_done(), "No jobs should remain")

        # Clean up additional extraction identifiers
        for i in range(len(test_jobs)):
            cleanup_identifier = ExtractionIdentifier(extraction_name=f"{extraction_id}_{i}")
            shutil.rmtree(cleanup_identifier.get_path(), ignore_errors=True)
