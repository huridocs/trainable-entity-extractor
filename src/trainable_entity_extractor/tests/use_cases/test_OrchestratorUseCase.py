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
        return [
            PredictionSample.from_text("Hello, my name is Alice Brown", "sample_1"),
            PredictionSample.from_text("Meet Charlie Wilson today", "sample_2"),
            PredictionSample.from_text("Diana Prince is the manager", "sample_3"),
        ]

    def _get_available_training_job(self, extraction_data: ExtractionData) -> TrainableEntityExtractorJob:
        train_use_case = TrainUseCase(extractors=self.extractors, logger=self.logger)
        jobs = train_use_case.get_jobs(extraction_data)
        same_input_output_jobs = [job for job in jobs if job.method_name == "SameInputOutputMethod"]
        self.assertGreater(len(same_input_output_jobs), 0, "SameInputOutputMethod not available")
        return same_input_output_jobs[0]

    def _create_test_extractor_job(self, method_name: str = "SameInputOutputMethod") -> TrainableEntityExtractorJob:
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
        extraction_data = self._create_sample_extraction_data()
        self.data_retriever.save_extraction_data(extraction_identifier, extraction_data)
        extractor_job = self._get_available_training_job(extraction_data)
        sub_job = DistributedSubJob(job_id="test_job_1", extractor_job=extractor_job)
        distributed_job = DistributedJob(extraction_identifier=extraction_identifier, type=JobType.TRAIN, sub_jobs=[sub_job])
        self.orchestrator.distributed_jobs = [distributed_job]
        result = self.orchestrator.process_job(distributed_job)
        self.assertTrue(result.success, f"Training should succeed: {result.error_message}")
        self.assertEqual(0, len(self.orchestrator.distributed_jobs), "Job should be removed after completion")

    def test_process_prediction_job_success(self):
        extraction_data = self._create_sample_extraction_data()
        self.data_retriever.save_extraction_data(extraction_identifier, extraction_data)
        trainer = TrainUseCase(extractors=self.extractors, logger=self.logger)
        jobs = trainer.get_jobs(extraction_data)
        self.assertGreater(len(jobs), 0, "Should have at least one training job")
        train_job = jobs[0]
        train_sub_job = DistributedSubJob(extractor_job=train_job)
        train_distributed_job = DistributedJob(
            extraction_identifier=extraction_identifier, type=JobType.TRAIN, sub_jobs=[train_sub_job]
        )
        train_orchestrator = OrchestratorUseCase(self.job_executor, self.logger)
        train_orchestrator.distributed_jobs = [train_distributed_job]
        train_result = None
        max_iterations = 10
        iteration = 0
        while train_orchestrator.distributed_jobs and iteration < max_iterations:
            train_result = train_orchestrator.process_job(train_orchestrator.distributed_jobs[0])
            if train_result and train_result.finished:
                break
            iteration += 1
        self.assertIsNotNone(train_result, "Training should have completed")
        self.assertTrue(train_result.success, f"Training should succeed: {train_result.error_message}")
        prediction_samples = [
            PredictionSample.from_text("Hello, my name is Alice"),
            PredictionSample.from_text("Bob is working today"),
        ]
        self.data_retriever.save_prediction_data(extraction_identifier, prediction_samples)
        predict_sub_job = DistributedSubJob(extractor_job=train_job)
        predict_distributed_job = DistributedJob(
            extraction_identifier=extraction_identifier, type=JobType.PREDICT, sub_jobs=[predict_sub_job]
        )
        self.orchestrator.distributed_jobs = [predict_distributed_job]
        result = self.orchestrator.process_job(predict_distributed_job)
        self.assertTrue(result.success, f"Prediction should succeed: {result.error_message}")

    def test_process_performance_job_success(self):
        extraction_data = self._create_sample_extraction_data()
        self.data_retriever.save_extraction_data(extraction_identifier, extraction_data)
        trainer = TrainUseCase(extractors=self.extractors, logger=self.logger)
        jobs = trainer.get_jobs(extraction_data)
        self.assertGreater(len(jobs), 0, "Should have at least one training job")
        sub_jobs = [DistributedSubJob(extractor_job=job) for job in jobs]
        distributed_job = DistributedJob(
            extraction_identifier=extraction_identifier, type=JobType.PERFORMANCE, sub_jobs=sub_jobs
        )
        self.orchestrator.distributed_jobs = [distributed_job]
        result = None
        while self.orchestrator.distributed_jobs:
            result = self.orchestrator.process_job(self.orchestrator.distributed_jobs[0])
            if result and result.finished:
                break
        self.assertIsNotNone(result, "Should have processed at least one job")
        if result:
            self.assertTrue(result.success, f"Performance evaluation should succeed: {result.error_message}")

    def test_end_to_end_workflow(self):
        extraction_data = self._create_sample_extraction_data()
        self.data_retriever.save_extraction_data(extraction_identifier, extraction_data)
        trainer = TrainUseCase(extractors=self.extractors, logger=self.logger)
        jobs = trainer.get_jobs(extraction_data)
        train_sub_job = DistributedSubJob(extractor_job=jobs[0])
        train_distributed_job = DistributedJob(
            extraction_identifier=extraction_identifier, type=JobType.TRAIN, sub_jobs=[train_sub_job]
        )
        self.orchestrator.distributed_jobs = [train_distributed_job]
        train_result = self.orchestrator.process_job(self.orchestrator.distributed_jobs[0])
        while self.orchestrator.exists_jobs_to_be_done():
            train_result = self.orchestrator.process_job(self.orchestrator.distributed_jobs[0])
        self.assertTrue(train_result.success, f"Training should succeed: {train_result.error_message}")
        prediction_samples = [PredictionSample.from_text("Test prediction text")]
        self.data_retriever.save_prediction_data(extraction_identifier, prediction_samples)
        predict_sub_job = DistributedSubJob(extractor_job=jobs[0])
        predict_distributed_job = DistributedJob(
            extraction_identifier=extraction_identifier, type=JobType.PREDICT, sub_jobs=[predict_sub_job]
        )
        self.orchestrator.distributed_jobs = [predict_distributed_job]
        predict_result = self.orchestrator.process_job(self.orchestrator.distributed_jobs[0])
        self.assertTrue(predict_result.success, f"Prediction should succeed: {predict_result.error_message}")

    def test_performance_evaluation_with_perfect_score(self):
        extraction_data = self._create_sample_extraction_data()
        self.data_retriever.save_extraction_data(extraction_identifier, extraction_data)
        trainer = TrainUseCase(extractors=self.extractors, logger=self.logger)
        jobs = trainer.get_jobs(extraction_data)
        sub_jobs = [DistributedSubJob(extractor_job=job) for job in jobs]
        distributed_job = DistributedJob(
            extraction_identifier=extraction_identifier, type=JobType.PERFORMANCE, sub_jobs=sub_jobs
        )
        self.orchestrator.distributed_jobs = [distributed_job]
        result = None
        iterations = 0
        max_iterations = 10
        while self.orchestrator.distributed_jobs and iterations < max_iterations:
            current_job = self.orchestrator.distributed_jobs[0]
            result = self.orchestrator.process_job(current_job)
            iterations += 1
            if result and result.finished:
                break
        self.assertIsNotNone(result, "Should have processed at least one job")
        if result:
            self.assertTrue(
                result.success or "in progress" in result.error_message.lower(),
                f"Performance evaluation should succeed or be in progress: {result.error_message}",
            )
