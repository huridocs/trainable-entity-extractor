from typing import List, Optional, Tuple

from trainable_entity_extractor.domain.JobPerformance import JobPerformance
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.ports.JobExecutor import JobExecutor
from trainable_entity_extractor.ports.ModelStorage import ModelStorage
from trainable_entity_extractor.use_cases.JobSelector import JobSelector
from trainable_entity_extractor.use_cases.send_logs import send_logs


class TrainingOrchestrator:
    def __init__(self, job_executor: JobExecutor, model_storage: ModelStorage):
        self.job_executor = job_executor
        self.model_storage = model_storage

    def process_performance_evaluation(
        self,
        extractor_jobs: List[TrainableEntityExtractorJob],
        options: list,
        multi_value: bool,
        extraction_identifier: ExtractionIdentifier,
    ) -> Tuple[bool, str, Optional[TrainableEntityExtractorJob]]:
        completed_jobs: List[JobPerformance] = []
        perfect_job: Optional[JobPerformance] = None

        for extractor_job in extractor_jobs:
            try:
                performance_result = self.job_executor.execute_performance_evaluation(extractor_job, options, multi_value)

                job_performance = JobPerformance(
                    extractor_job=extractor_job,
                    performance_result=performance_result,
                    job_id=f"perf_{extractor_job.method_name}",
                )

                completed_jobs.append(job_performance)

                if job_performance.is_perfect:
                    perfect_job = job_performance
                    send_logs(extraction_identifier, f"Perfect performance found for method {extractor_job.method_name}")
                    break

            except Exception as e:
                send_logs(
                    extraction_identifier,
                    f"Performance evaluation failed for method {extractor_job.method_name}: {e}",
                    LogSeverity.error,
                )

        selected_job = JobSelector.select_best_job(completed_jobs, len(extractor_jobs), perfect_job)

        if not selected_job:
            return False, "No suitable job found after performance evaluation", None

        if selected_job.should_be_retrained_with_more_data:
            send_logs(extraction_identifier, f"Job {selected_job.method_name} needs retraining with more data")
            return self._handle_retraining(selected_job, options, multi_value, extraction_identifier)

        success = self._upload_model_with_completion_signal(extraction_identifier, selected_job)
        if success:
            return True, f"Model uploaded successfully for method {selected_job.method_name}", selected_job
        else:
            return False, "Model upload failed", None

    def process_single_training(
        self,
        extractor_job: TrainableEntityExtractorJob,
        options: list,
        multi_value: bool,
        extraction_identifier: ExtractionIdentifier,
    ) -> Tuple[bool, str]:
        try:
            success, error_message = self.job_executor.execute_training(extractor_job, options, multi_value)

            if not success:
                send_logs(extraction_identifier, f"Training failed: {error_message}", LogSeverity.error)
                return False, error_message

            upload_success = self._upload_model_with_completion_signal(extraction_identifier, extractor_job)
            if upload_success:
                send_logs(
                    extraction_identifier,
                    f"Training and upload completed successfully for method {extractor_job.method_name}",
                )
                return True, "Training completed successfully"
            else:
                return False, "Training succeeded but model upload failed"

        except Exception as e:
            error_msg = f"Training failed with exception: {e}"
            send_logs(extraction_identifier, error_msg, LogSeverity.error)
            return False, error_msg

    def _handle_retraining(
        self,
        extractor_job: TrainableEntityExtractorJob,
        options: list,
        multi_value: bool,
        extraction_identifier: ExtractionIdentifier,
    ) -> Tuple[bool, str, Optional[TrainableEntityExtractorJob]]:
        try:
            success, error_message = self.job_executor.execute_training(extractor_job, options, multi_value)

            if success:
                upload_success = self._upload_model_with_completion_signal(extraction_identifier, extractor_job)
                if upload_success:
                    return True, "Retraining completed successfully", extractor_job
                else:
                    return False, "Retraining succeeded but model upload failed", None
            else:
                return False, f"Retraining failed: {error_message}", None

        except Exception as e:
            return False, f"Retraining failed with exception: {e}", None

    def _upload_model_with_completion_signal(
        self, extraction_identifier: ExtractionIdentifier, extractor_job: TrainableEntityExtractorJob
    ) -> bool:
        try:
            upload_success = self.model_storage.upload_model(extraction_identifier, extractor_job.method_name, extractor_job)

            if upload_success:
                signal_success = self.model_storage.create_model_completion_signal(extraction_identifier)
                if signal_success:
                    send_logs(
                        extraction_identifier, f"Model and completion signal uploaded for method {extractor_job.method_name}"
                    )
                    return True
                else:
                    send_logs(
                        extraction_identifier,
                        f"Model uploaded but completion signal creation failed for method {extractor_job.method_name}",
                        LogSeverity.error,
                    )
                    return False
            else:
                send_logs(
                    extraction_identifier, f"Model upload failed for method {extractor_job.method_name}", LogSeverity.error
                )
                return False

        except Exception as e:
            send_logs(extraction_identifier, f"Model upload failed with exception: {e}", LogSeverity.error)
            return False
