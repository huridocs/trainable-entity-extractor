from typing import Tuple
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.ports.JobExecutor import JobExecutor
from trainable_entity_extractor.ports.ModelStorage import ModelStorage
from trainable_entity_extractor.use_cases.send_logs import send_logs


class PredictionOrchestrator:
    def __init__(self, job_executor: JobExecutor, model_storage: ModelStorage):
        self.job_executor = job_executor
        self.model_storage = model_storage
        self.max_model_wait_retries = 10
        self.current_retry_count = 0

    def process_prediction(
        self,
        extractor_job: TrainableEntityExtractorJob,
        extraction_identifier: ExtractionIdentifier,
        wait_for_model: bool = True,
    ) -> Tuple[bool, str, bool]:
        if wait_for_model:
            model_available = self._check_and_wait_for_model(extraction_identifier)
            if not model_available:
                if self.current_retry_count < self.max_model_wait_retries:
                    self.current_retry_count += 1
                    send_logs(
                        extraction_identifier,
                        f"Model not ready yet, will retry (attempt {self.current_retry_count}/{self.max_model_wait_retries})",
                    )
                    return False, "Model not ready, retrying", True
                else:
                    send_logs(extraction_identifier, "Model not available after maximum retries", LogSeverity.error)
                    return False, "Model not available after maximum wait time", False

        try:
            success, error_message = self.job_executor.execute_prediction(extractor_job)

            if success:
                send_logs(extraction_identifier, f"Prediction completed successfully for method {extractor_job.method_name}")
                return True, "Prediction completed successfully", False
            else:
                send_logs(extraction_identifier, f"Prediction failed: {error_message}", LogSeverity.error)
                return False, error_message, False

        except Exception as e:
            error_msg = f"Prediction failed with exception: {e}"
            send_logs(extraction_identifier, error_msg, LogSeverity.error)
            return False, error_msg, False

    def _check_and_wait_for_model(self, extraction_identifier: ExtractionIdentifier) -> bool:
        try:
            model_downloaded = self.model_storage.download_model(extraction_identifier)
            if not model_downloaded:
                send_logs(extraction_identifier, "Model download failed, checking completion signal", LogSeverity.warning)

            # Check for completion signal file
            completion_signal_exists = self.model_storage.check_model_completion_signal(extraction_identifier)

            if completion_signal_exists:
                send_logs(extraction_identifier, "Model completion signal found, model is ready")
                return True
            else:
                send_logs(extraction_identifier, "Model completion signal not found, model may still be uploading")
                return False

        except Exception as e:
            send_logs(extraction_identifier, f"Error checking model availability: {e}", LogSeverity.error)
            return False

    def reset_retry_count(self):
        """Reset the retry count for a new prediction job"""
        self.current_retry_count = 0
