from typing import Tuple, List
from trainable_entity_extractor.domain.DistributedJob import DistributedJob
from trainable_entity_extractor.domain.DistributedSubJob import DistributedSubJob
from trainable_entity_extractor.domain.JobStatus import JobStatus
from trainable_entity_extractor.domain.JobType import JobType
from trainable_entity_extractor.domain.PerformanceSummary import PerformanceSummary
from trainable_entity_extractor.domain.JobProcessingResult import JobProcessingResult
from trainable_entity_extractor.ports.JobExecutor import JobExecutor
from trainable_entity_extractor.ports.Logger import Logger
from trainable_entity_extractor.use_cases.JobSelectorUseCase import JobSelectorUseCase


class OrchestratorUseCase:
    def __init__(self, job_executor: JobExecutor, logger: Logger, distributed_jobs: List[DistributedJob] = None):
        self.job_executor = job_executor
        self.logger = logger
        self.distributed_jobs: List[DistributedJob] = distributed_jobs or []

    def add_job(self, distributed_job: DistributedJob) -> None:
        self.distributed_jobs.append(distributed_job)

    def process_job(self, distributed_job: DistributedJob) -> JobProcessingResult:
        if self.job_executor.is_extractor_cancelled(distributed_job.extraction_identifier):
            self._cancel_and_remove_job(distributed_job)
            return JobProcessingResult(
                finished=True,
                success=False,
                error_message=f"Job cancelled for extraction {distributed_job.extraction_identifier}",
            )

        if distributed_job.type == JobType.TRAIN:
            return self._process_training_job(distributed_job)
        elif distributed_job.type == JobType.PREDICT:
            return self._process_prediction_job(distributed_job)
        elif distributed_job.type == JobType.PERFORMANCE:
            return self._process_performance_job(distributed_job)
        else:
            self.distributed_jobs.remove(distributed_job)
            return JobProcessingResult(
                finished=True, success=False, error_message=f"Unknown job type: {distributed_job.type}"
            )

    def _cancel_and_remove_job(self, distributed_job: DistributedJob) -> None:
        self.job_executor.cancel_jobs(distributed_job)
        self._remove_job_from_queue(distributed_job)

    def _process_training_job(self, distributed_job: DistributedJob) -> JobProcessingResult:
        extraction_identifier = distributed_job.extraction_identifier
        sub_job = distributed_job.sub_jobs[0]

        if sub_job.status == JobStatus.WAITING:
            self.job_executor.start_training(extraction_identifier, sub_job)

        if sub_job.status == JobStatus.SUCCESS:
            self.distributed_jobs.remove(distributed_job)
            if self.job_executor.upload_model(extraction_identifier, sub_job.extractor_job):
                return JobProcessingResult(
                    finished=True,
                    success=True,
                    error_message=f"Training completed successfully for method {sub_job.extractor_job.method_name}",
                    gpu_needed=getattr(sub_job.extractor_job, "requires_gpu", False),
                )
            else:
                return JobProcessingResult(
                    finished=True, success=False, error_message="Training completed but model upload failed"
                )
        elif sub_job.status == JobStatus.FAILURE:
            self.distributed_jobs.remove(distributed_job)
            return JobProcessingResult(
                finished=True, success=False, error_message=f"Training failed for method {sub_job.extractor_job.method_name}"
            )
        else:
            return JobProcessingResult(
                finished=False,
                success=False,
                error_message="Training in progress",
                gpu_needed=getattr(sub_job.extractor_job, "requires_gpu", False),
            )

    def _process_prediction_job(self, distributed_job: DistributedJob) -> JobProcessingResult:
        extraction_identifier = distributed_job.extraction_identifier
        sub_job = distributed_job.sub_jobs[0]

        if sub_job.status == JobStatus.WAITING:
            self.job_executor.start_prediction(extraction_identifier, sub_job)

        if sub_job.status == JobStatus.SUCCESS:
            self.distributed_jobs.remove(distributed_job)
            if sub_job.result:
                return JobProcessingResult(
                    finished=True,
                    success=True,
                    error_message=f"Prediction completed successfully for method {sub_job.extractor_job.method_name}",
                    gpu_needed=getattr(sub_job.extractor_job, "requires_gpu", False),
                )
            else:
                return JobProcessingResult(
                    finished=True, success=False, error_message="Prediction completed but no results generated"
                )
        elif sub_job.status == JobStatus.FAILURE:
            self.distributed_jobs.remove(distributed_job)
            return JobProcessingResult(
                finished=True,
                success=False,
                error_message=f"Prediction failed for method {sub_job.extractor_job.method_name}",
            )
        else:
            return JobProcessingResult(
                finished=False,
                success=False,
                error_message="Prediction in progress",
                gpu_needed=getattr(sub_job.extractor_job, "requires_gpu", False),
            )

    def _process_performance_job(self, distributed_job: DistributedJob) -> JobProcessingResult:
        self.job_executor.update_job_statuses(distributed_job)

        if len(distributed_job.sub_jobs) == [x for x in distributed_job.sub_jobs if x.status == JobStatus.WAITING]:
            self.job_executor.recreate_model_folder(distributed_job.extraction_identifier)

        self._start_pending_performance_evaluations(distributed_job)

        if self._has_perfect_score_job(distributed_job):
            self.job_executor.cancel_jobs(distributed_job)

        if not self._are_all_jobs_complete(distributed_job):
            return JobProcessingResult(
                finished=False,
                success=False,
                error_message="Performance evaluation in progress",
                gpu_needed=any(getattr(job.extractor_job, "requires_gpu", False) for job in distributed_job.sub_jobs),
            )

        self._log_performance_summary(distributed_job)
        self._remove_job_from_queue(distributed_job)

        return self._handle_performance_results(distributed_job)

    def _start_pending_performance_evaluations(self, distributed_job: DistributedJob) -> None:
        for sub_job in distributed_job.sub_jobs:
            if self.job_executor.is_extractor_cancelled(distributed_job.extraction_identifier):
                self._cancel_and_remove_job(distributed_job)
                return

            if sub_job.status == JobStatus.WAITING:
                self.job_executor.start_performance_evaluation(distributed_job.extraction_identifier, sub_job)

            if sub_job.result and hasattr(sub_job.result, "is_perfect") and sub_job.result.is_perfect:
                break

    @staticmethod
    def _has_perfect_score_job(distributed_job: DistributedJob) -> bool:
        perfect_score_jobs = [
            job
            for job in distributed_job.sub_jobs
            if job.result and hasattr(job.result, "is_perfect") and job.result.is_perfect
        ]
        return len(perfect_score_jobs) > 0

    def _are_all_jobs_complete(self, distributed_job: DistributedJob) -> bool:
        return all(sub_job.status in self.job_executor.get_finished_status() for sub_job in distributed_job.sub_jobs)

    def _log_performance_summary(self, distributed_job: DistributedJob) -> None:
        performance_summary = PerformanceSummary.from_distributed_job(distributed_job)

        for sub_job in distributed_job.sub_jobs:
            if sub_job.status == JobStatus.SUCCESS and sub_job.result:
                performance_summary.add_performance_from_sub_job(sub_job)

        summary_log = performance_summary.to_log()
        self.logger.log(distributed_job.extraction_identifier, summary_log)

    def _remove_job_from_queue(self, distributed_job: DistributedJob) -> None:
        if distributed_job in self.distributed_jobs:
            self.distributed_jobs.remove(distributed_job)

    def _handle_performance_results(self, distributed_job: DistributedJob) -> JobProcessingResult:
        best_job: DistributedSubJob = JobSelectorUseCase.select_best_job(distributed_job)

        if not best_job:
            return JobProcessingResult(
                finished=True, success=False, error_message="No valid performance results to select the best model"
            )

        if not best_job.extractor_job.should_be_retrained_with_more_data:
            return self._finalize_best_model(distributed_job, best_job)
        else:
            return self._schedule_retraining(distributed_job, best_job)

    def _finalize_best_model(self, distributed_job: DistributedJob, best_job: DistributedSubJob) -> JobProcessingResult:
        if self.job_executor.upload_model(distributed_job.extraction_identifier, best_job.extractor_job):
            performance_score = self._extract_performance_score(best_job)
            return JobProcessingResult(
                finished=True,
                success=True,
                error_message=f"Best model selected: {best_job.extractor_job.method_name} with performance {performance_score}",
                gpu_needed=getattr(best_job.extractor_job, "requires_gpu", False),
            )
        else:
            return JobProcessingResult(finished=True, success=False, error_message="Best model selected but upload failed")

    def _schedule_retraining(self, distributed_job: DistributedJob, best_job: DistributedSubJob) -> JobProcessingResult:
        training_job = DistributedJob(
            extraction_identifier=distributed_job.extraction_identifier,
            type=JobType.TRAIN,
            sub_jobs=[
                DistributedSubJob(
                    job_id=f"retrain_{best_job.extractor_job.method_name}", extractor_job=best_job.extractor_job
                )
            ],
        )
        self.distributed_jobs.append(training_job)
        return JobProcessingResult(
            finished=False,
            success=False,
            error_message="Retraining model",
            gpu_needed=getattr(best_job.extractor_job, "requires_gpu", False),
        )

    @staticmethod
    def _extract_performance_score(best_job: DistributedSubJob) -> str:
        if best_job.result and hasattr(best_job.result, "performance_score"):
            return str(best_job.result.performance_score)
        else:
            return "unknown"

    def exists_jobs_to_be_done(self) -> bool:
        return len(self.distributed_jobs) > 0

    def execute_job_for_domain(self, domain: str) -> tuple[JobProcessingResult, DistributedJob | None]:
        for job in self.distributed_jobs:
            if job.domain_name == domain:
                return self.process_job(job), job

        return JobProcessingResult(finished=False, success=False, error_message=f"No job found for domain {domain}"), None
