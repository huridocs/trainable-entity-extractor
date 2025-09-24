import shutil

from trainable_entity_extractor.adapters.ExtractorLogger import ExtractorLogger
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.PdfToMultiOptionExtractor import (
    PdfToMultiOptionExtractor,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_text_extractor.PdfToTextExtractor import PdfToTextExtractor
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.TextToMultiOptionExtractor import (
    TextToMultiOptionExtractor,
)
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.TextToTextExtractor import TextToTextExtractor
from trainable_entity_extractor.domain.DistributedJob import DistributedJob
from trainable_entity_extractor.domain.DistributedSubJob import DistributedSubJob
from trainable_entity_extractor.domain.JobType import JobType
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.adapters.LocalJobExecutor import LocalJobExecutor
from trainable_entity_extractor.adapters.LocalModelStorage import LocalModelStorage
from trainable_entity_extractor.adapters.LocalExtractionDataRetriever import LocalExtractionDataRetriever
from trainable_entity_extractor.ports.ExtractorBase import ExtractorBase
from trainable_entity_extractor.use_cases.OrchestratorUseCase import OrchestratorUseCase
from trainable_entity_extractor.use_cases.TrainUseCase import TrainUseCase


class TrainableEntityExtractor:
    EXTRACTORS: list[type[ExtractorBase]] = [
        PdfToMultiOptionExtractor,
        TextToMultiOptionExtractor,
        PdfToTextExtractor,
        TextToTextExtractor,
    ]

    def __init__(self, extraction_identifier: ExtractionIdentifier):
        self.extraction_identifier = extraction_identifier
        self.multi_value: bool = False
        self.options: list = list()
        self.data_retriever = LocalExtractionDataRetriever()
        self.model_storage = LocalModelStorage()
        self.logger = ExtractorLogger()
        self.job_executor = LocalJobExecutor(self.EXTRACTORS, self.data_retriever, self.model_storage, self.logger)

    def train(self, extraction_data: ExtractionData) -> tuple[bool, str]:
        if not self._is_training_valid(extraction_data):
            return False, "Training validation failed"

        self.data_retriever.save_extraction_data(self.extraction_identifier, extraction_data)

        jobs = self._get_training_jobs(extraction_data)
        if not jobs:
            return False, "No suitable extractors found for training"

        return self._execute_training_workflow(jobs)

    def predict(self, prediction_samples: list[PredictionSample]) -> list[Suggestion]:
        if not self._is_prediction_valid(prediction_samples):
            return []

        self.data_retriever.save_prediction_data(self.extraction_identifier, prediction_samples)
        extractor_job = self._get_extractor_job()
        if not extractor_job:
            return []

        self._execute_prediction(extractor_job)
        return self.data_retriever.get_suggestions(self.extraction_identifier)

    def _is_training_valid(self, extraction_data: ExtractionData) -> bool:
        if extraction_data.extraction_identifier.is_training_canceled():
            self.logger.log(self.extraction_identifier, "Training canceled", LogSeverity.error)
            return False

        if not extraction_data or not extraction_data.samples:
            return False

        return True

    def _get_training_jobs(self, extraction_data: ExtractionData) -> list:
        trainer = TrainUseCase(extractors=self.EXTRACTORS)
        jobs = trainer.get_jobs(extraction_data)

        if not jobs:
            self.logger.log(self.extraction_identifier, "No suitable extractors found for training", LogSeverity.error)

        return jobs

    def _create_distributed_training_jobs(self, jobs: list) -> list[DistributedJob]:
        return [
            DistributedJob(
                extraction_identifier=self.extraction_identifier,
                type=JobType.TRAIN,
                sub_jobs=[DistributedSubJob(extractor_job=job) for job in jobs],
            )
        ]

    def _execute_training_workflow(self, jobs: list) -> tuple[bool, str]:
        distributed_jobs = self._create_distributed_training_jobs(jobs)
        training_orchestrator = OrchestratorUseCase(self.job_executor, self.logger, distributed_jobs)

        self.logger.log(self.extraction_identifier, f"Training with {len(jobs)} available methods")

        try:
            success, message = self._process_training_jobs(training_orchestrator, distributed_jobs)
            return self._finalize_training(success, message)
        except Exception as e:
            return self._handle_training_exception(e)

    @staticmethod
    def _process_training_jobs(training_orchestrator: OrchestratorUseCase, distributed_jobs: list) -> tuple[bool, str]:
        success = False
        message = "Unknown error during training"

        while training_orchestrator.exists_jobs_to_be_done():
            success, message = training_orchestrator.process_job(distributed_jobs[0])

        return success, message

    def _finalize_training(self, success: bool, message: str) -> tuple[bool, str]:
        if success:
            self.logger.log(self.extraction_identifier, f"Training completed successfully: {message}")
        else:
            self.logger.log(self.extraction_identifier, f"Training failed: {message}", LogSeverity.error)

        self.extraction_identifier.clean_extractor_folder()
        return success, message

    def _handle_training_exception(self, exception: Exception) -> tuple[bool, str]:
        error_message = f"Training failed with exception: {str(exception)}"
        self.logger.log(self.extraction_identifier, error_message, LogSeverity.error)
        shutil.rmtree(self.extraction_identifier.get_path(), ignore_errors=True)
        return False, error_message

    @staticmethod
    def _is_prediction_valid(prediction_samples: list[PredictionSample]) -> bool:
        return bool(prediction_samples)

    def _get_extractor_job(self):
        extractor_job = self.model_storage.get_extractor_job(self.extraction_identifier)

        if not extractor_job:
            self.logger.log(self.extraction_identifier, "No trained method found for prediction", LogSeverity.error)

        return extractor_job

    def _execute_prediction(self, extractor_job) -> None:
        distributed_jobs = self._create_distributed_prediction_jobs(extractor_job)
        prediction_orchestrator = OrchestratorUseCase(self.job_executor, self.logger)
        prediction_orchestrator.distributed_jobs = distributed_jobs

        self.logger.log(self.extraction_identifier, f"Predicting using method {extractor_job.method_name}")

        try:
            self._process_prediction_jobs(prediction_orchestrator, distributed_jobs)
        except Exception as e:
            self._handle_prediction_exception(e)

    def _create_distributed_prediction_jobs(self, extractor_job) -> list[DistributedJob]:
        return [
            DistributedJob(
                extraction_identifier=self.extraction_identifier,
                type=JobType.PREDICT,
                sub_jobs=[DistributedSubJob(extractor_job=extractor_job)],
            )
        ]

    def _process_prediction_jobs(self, prediction_orchestrator: OrchestratorUseCase, distributed_jobs: list) -> None:
        while prediction_orchestrator.exists_jobs_to_be_done():
            success, message = prediction_orchestrator.process_job(distributed_jobs[0])

            if success:
                self.logger.log(self.extraction_identifier, f"Prediction completed: {message}")
                break
            elif "in progress" not in message.lower():
                self.logger.log(self.extraction_identifier, f"Prediction failed: {message}", LogSeverity.error)
                break

    def _log_prediction_success(self, suggestions: list[Suggestion], method_name: str):
        self.logger.log(self.extraction_identifier, f"Generated {len(suggestions)} suggestions using {method_name}")

    def _handle_prediction_exception(self, exception: Exception):
        error_message = f"Prediction failed with exception: {str(exception)}"
        self.logger.log(self.extraction_identifier, error_message, LogSeverity.error)
