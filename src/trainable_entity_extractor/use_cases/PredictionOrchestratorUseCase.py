from typing import Tuple, List
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.ports.JobExecutor import JobExecutor
from trainable_entity_extractor.ports.Logger import Logger


class PredictionOrchestratorUseCase:
    def __init__(self, job_executor: JobExecutor, logger: Logger):
        self.job_executor = job_executor
        self.max_model_wait_retries = 10
        self.current_retry_count = 0
        self.logger = logger

    def process_prediction(
        self,
        extractor_job: TrainableEntityExtractorJob,
        extraction_identifier: ExtractionIdentifier,
        wait_for_model: bool = True,
    ) -> Tuple[bool, str, bool]:
        if wait_for_model:
            model_available = self.job_executor.check_and_wait_for_model(extraction_identifier)
            if not model_available:
                if self.current_retry_count < self.max_model_wait_retries:
                    self.current_retry_count += 1
                    self.logger.log(
                        extraction_identifier,
                        f"Model not ready yet, will retry (attempt {self.current_retry_count}/{self.max_model_wait_retries})",
                    )
                    return False, "Model not ready, retrying", True
                else:
                    self.logger.log(extraction_identifier, "Model not available after maximum retries", LogSeverity.error)
                    return False, "Model not available after maximum wait time", False

        try:
            success, error_message = self.job_executor.execute_prediction(extractor_job)

            if success:
                self.logger.log(
                    extraction_identifier, f"Prediction completed successfully for method {extractor_job.method_name}"
                )
                return True, "Prediction completed successfully", False
            else:
                self.logger.log(extraction_identifier, f"Prediction failed: {error_message}", LogSeverity.error)
                return False, error_message, False

        except Exception as e:
            error_msg = f"Prediction failed with exception: {e}"
            self.logger.log(extraction_identifier, error_msg, LogSeverity.error)
            return False, error_msg, False

    def reset_retry_count(self):
        """Reset the retry count for a new prediction job"""
        self.current_retry_count = 0

    def process_prediction_with_samples(
        self,
        extractor_job: TrainableEntityExtractorJob,
        prediction_samples: List[PredictionSample],
        extraction_identifier: ExtractionIdentifier,
        wait_for_model: bool = True,
    ) -> Tuple[bool, str, List[Suggestion]]:
        """Process prediction with samples and return actual suggestions"""
        if wait_for_model:
            model_available = self.job_executor.check_and_wait_for_model(extraction_identifier)
            if not model_available:
                if self.current_retry_count < self.max_model_wait_retries:
                    self.current_retry_count += 1
                    self.logger.log(
                        extraction_identifier,
                        f"Model not ready yet, will retry (attempt {self.current_retry_count}/{self.max_model_wait_retries})",
                    )
                    return False, "Model not ready, retrying", []
                else:
                    self.logger.log(extraction_identifier, "Model not available after maximum retries", LogSeverity.error)
                    return False, "Model not available after maximum wait time", []

        try:
            # Check if job executor has the execute_prediction_with_samples method
            if hasattr(self.job_executor, "execute_prediction_with_samples"):
                success, error_message, suggestions = self.job_executor.execute_prediction_with_samples(
                    extractor_job, prediction_samples
                )

                if success:
                    self.logger.log(
                        extraction_identifier,
                        f"Prediction with samples completed successfully for method {extractor_job.method_name}",
                    )
                    return True, "Prediction completed successfully", suggestions
                else:
                    self.logger.log(extraction_identifier, f"Prediction failed: {error_message}", LogSeverity.error)
                    return False, error_message, []
            else:
                # Fallback to regular prediction
                success, error_message = self.job_executor.execute_prediction(extractor_job)

                if success:
                    self.logger.log(
                        extraction_identifier,
                        f"Prediction completed successfully for method {extractor_job.method_name}",
                    )
                    return True, "Prediction completed successfully", []
                else:
                    self.logger.log(extraction_identifier, f"Prediction failed: {error_message}", LogSeverity.error)
                    return False, error_message, []

        except Exception as e:
            error_msg = f"Prediction failed with exception: {e}"
            self.logger.log(extraction_identifier, error_msg, LogSeverity.error)
            return False, error_msg, []
