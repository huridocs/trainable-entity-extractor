from unittest import TestCase
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.services.EntityExtractionService import EntityExtractionService
from trainable_entity_extractor.adapters.LocalJobExecutor import LocalJobExecutor
from trainable_entity_extractor.adapters.LocalModelStorage import LocalModelStorage
from trainable_entity_extractor.use_cases.TrainingOrchestrator import TrainingOrchestratorUseCase
from trainable_entity_extractor.use_cases.PredictionOrchestrator import PredictionOrchestrator
from trainable_entity_extractor.drivers.TrainableEntityExtractor import TrainableEntityExtractor


class TestOrchestratorRefactoring(TestCase):
    """
    Test class demonstrating the new orchestrator approach that replaces
    the old create_model and get_suggestions methods in extractors.
    """

    def setUp(self):
        self.tenant = "unit_test"
        self.extraction_id = "TestOrchestratorRefactoring"
        self.extraction_identifier = ExtractionIdentifier(run_name=self.tenant, extraction_name=self.extraction_id)

        # Create sample training data
        self.extraction_data = ExtractionData(
            samples=[
                TrainingSample(labeled_data=LabeledData(label_text="Sample 1")),
                TrainingSample(labeled_data=LabeledData(label_text="Sample 2")),
                TrainingSample(labeled_data=LabeledData(label_text="Sample 3")),
            ],
            extraction_identifier=self.extraction_identifier,
            options=["Option A", "Option B"],
            multi_value=False,
        )

    def test_distributed_performance_creation(self):
        """Test that DistributedPerformance is correctly created and used"""
        extraction_service = EntityExtractionService(self.extraction_identifier)
        trainable_extractor = TrainableEntityExtractor(self.extraction_identifier)

        # Get distributed jobs
        jobs = trainable_extractor.get_jobs(self.extraction_data)
        self.assertGreater(len(jobs), 0)

        # Test that should_be_retrained_with_more_data is properly set in jobs
        for job in jobs:
            self.assertIsInstance(job.should_be_retrained_with_more_data, bool)

        # Test get_performance returns DistributedPerformance
        if jobs:
            distributed_performance = trainable_extractor.get_performance(jobs[0], self.extraction_data)
            self.assertIsNotNone(distributed_performance.method_name)
            self.assertIsInstance(distributed_performance.performance, float)
            self.assertIsInstance(distributed_performance.execution_seconds, int)
            self.assertIn(distributed_performance.status, ["pending", "running", "completed", "failed"])

    def test_train_one_method_returns_tuple(self):
        """Test that train_one_method returns (bool, str) tuple"""
        trainable_extractor = TrainableEntityExtractor(self.extraction_identifier)
        jobs = trainable_extractor.get_jobs(self.extraction_data)

        if jobs:
            success, message = trainable_extractor.train_one_method(jobs[0], self.extraction_data)
            self.assertIsInstance(success, bool)
            self.assertIsInstance(message, str)

    def test_job_executor_interface_implementation(self):
        """Test that LocalJobExecutor properly implements the JobExecutor interface"""
        trainable_extractor = TrainableEntityExtractor(self.extraction_identifier)
        job_executor = LocalJobExecutor(trainable_extractor, self.extraction_data)

        jobs = trainable_extractor.get_jobs(self.extraction_data)
        if jobs:
            job = jobs[0]

            # Test execute_performance_evaluation returns DistributedPerformance
            distributed_perf = job_executor.start_performance_evaluation(job, [], False)
            self.assertIsNotNone(distributed_perf)
            self.assertEqual(distributed_perf.method_name, job.method_name)

            # Test execute_training returns (bool, str)
            success, message = job_executor.start_training(job, [], False)
            self.assertIsInstance(success, bool)
            self.assertIsInstance(message, str)

            # Test job status tracking
            job_id = f"test_job_{job.method_name}"
            job_executor.job_statuses[job_id] = "running"
            status = job_executor.get_job_status(job_id)
            self.assertEqual(status, "running")

    def test_model_storage_interface_implementation(self):
        """Test that LocalModelStorage properly implements the ModelStorage interface"""
        model_storage = LocalModelStorage()
        jobs = TrainableEntityExtractor(self.extraction_identifier).get_jobs(self.extraction_data)

        if jobs:
            job = jobs[0]

            # Test upload_model
            success = model_storage.upload_model(self.extraction_identifier, job.method_name, job)
            self.assertIsInstance(success, bool)

            # Test create_model_completion_signal
            signal_success = model_storage.create_model_completion_signal(self.extraction_identifier)
            self.assertIsInstance(signal_success, bool)

            # Test check_model_completion_signal
            if signal_success:
                check_result = model_storage.check_model_completion_signal(self.extraction_identifier)
                self.assertTrue(check_result)

    def test_training_orchestrator_with_new_interfaces(self):
        """Test TrainingOrchestrator with the new interface implementations"""
        trainable_extractor = TrainableEntityExtractor(self.extraction_identifier)
        job_executor = LocalJobExecutor(trainable_extractor, self.extraction_data)
        model_storage = LocalModelStorage()

        training_orchestrator = TrainingOrchestratorUseCase(job_executor, model_storage)
        jobs = trainable_extractor.get_jobs(self.extraction_data)

        if jobs:
            # Test single training process
            success, message = training_orchestrator.process_single_training(jobs[0], [], False, self.extraction_identifier)
            self.assertIsInstance(success, bool)
            self.assertIsInstance(message, str)

    def test_prediction_orchestrator_with_new_interfaces(self):
        """Test PredictionOrchestrator with the new interface implementations"""
        trainable_extractor = TrainableEntityExtractor(self.extraction_identifier)
        job_executor = LocalJobExecutor(trainable_extractor, self.extraction_data)
        model_storage = LocalModelStorage()

        prediction_orchestrator = PredictionOrchestrator(job_executor, model_storage)
        jobs = trainable_extractor.get_jobs(self.extraction_data)

        if jobs:
            # Test prediction process
            success, message, should_retry = prediction_orchestrator.process_prediction(
                jobs[0], self.extraction_identifier, wait_for_model=False
            )
            self.assertIsInstance(success, bool)
            self.assertIsInstance(message, str)
            self.assertIsInstance(should_retry, bool)

    def test_entity_extraction_service_complete_workflow(self):
        """Test the complete workflow using EntityExtractionService"""
        extraction_service = EntityExtractionService(self.extraction_identifier)

        # Test getting available jobs
        available_jobs = extraction_service.get_available_jobs(self.extraction_data)
        self.assertGreater(len(available_jobs), 0)

        # Test that should_be_retrained_with_more_data is properly populated
        for job in available_jobs:
            self.assertIsInstance(job.should_be_retrained_with_more_data, bool)

        # Test performance evaluation
        if available_jobs:
            success, message, performance = extraction_service.evaluate_method_performance(
                available_jobs[0], self.extraction_data
            )
            self.assertIsInstance(success, bool)
            self.assertIsInstance(message, str)
            self.assertIsInstance(performance, float)

        # Test training workflow
        success, message, selected_job = extraction_service.train_with_orchestrator(
            self.extraction_data, [], False, use_performance_evaluation=False
        )
        self.assertIsInstance(success, bool)
        self.assertIsInstance(message, str)

        if success and selected_job:
            # Test prediction workflow
            prediction_samples = [PredictionSample()]
            pred_success, pred_message, suggestions = extraction_service.predict_with_orchestrator(
                prediction_samples, selected_job, wait_for_model=False
            )
            self.assertIsInstance(pred_success, bool)
            self.assertIsInstance(pred_message, str)
            self.assertIsInstance(suggestions, list)

    def test_job_performance_uses_distributed_performance(self):
        """Test that JobPerformance correctly uses DistributedPerformance"""
        from trainable_entity_extractor.domain.JobPerformance import JobPerformance
        from trainable_entity_extractor.domain.DistributedPerformance import DistributedPerformance

        trainable_extractor = TrainableEntityExtractor(self.extraction_identifier)
        jobs = trainable_extractor.get_jobs(self.extraction_data)

        if jobs:
            job = jobs[0]
            distributed_perf = trainable_extractor.get_performance(job, self.extraction_data)

            job_performance = JobPerformance(extractor_job=job, performance_result=distributed_perf, job_id="test_job_123")

            # Test that properties work correctly with DistributedPerformance
            self.assertEqual(job_performance.performance_score, distributed_perf.performance)
            self.assertEqual(job_performance.method_name, job.method_name)
            self.assertEqual(job_performance.should_be_retrained_with_more_data, job.should_be_retrained_with_more_data)
            self.assertIsInstance(job_performance.is_perfect, bool)
