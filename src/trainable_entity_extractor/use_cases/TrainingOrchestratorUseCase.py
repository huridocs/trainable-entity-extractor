from typing import List, Tuple

from trainable_entity_extractor.domain.DistributedJob import DistributedJob
from trainable_entity_extractor.domain.DistributedSubJob import DistributedSubJob
from trainable_entity_extractor.domain.JobStatus import JobStatus

from trainable_entity_extractor.domain.JobType import JobType
from trainable_entity_extractor.ports.JobExecutor import JobExecutor
from trainable_entity_extractor.ports.Logger import Logger
from trainable_entity_extractor.use_cases.JobSelectorUseCase import JobSelectorUseCase


class TrainingOrchestratorUseCase:

    def __init__(self, distributed_jobs: List[DistributedJob], job_executor: JobExecutor, logger: Logger):
        self.job_executor = job_executor
        self.distributed_jobs: List[DistributedJob] = distributed_jobs
        self.logger = logger

    def process_job(self, distributed_job: DistributedJob) -> Tuple[bool, str]:
        if distributed_job.type == JobType.TRAIN:
            return self._process_training_job(distributed_job)
        elif distributed_job.type == JobType.PERFORMANCE:
            return self._process_performance_job(distributed_job)
        else:
            self.distributed_jobs.remove(distributed_job)
            return False, f"Unknown job type: {distributed_job.type}"

    def _process_training_job(self, distributed_job: DistributedJob) -> Tuple[bool, str]:
        extraction_identifier = distributed_job.extraction_identifier

        sub_job = distributed_job.sub_jobs[0]

        if sub_job.status == JobStatus.WAITING:
            self.job_executor.start_training(extraction_identifier, sub_job)

        if sub_job.status == JobStatus.SUCCESS:
            self.distributed_jobs.remove(distributed_job)
            if self.job_executor.upload_model(extraction_identifier, sub_job.extractor_job):
                return True, f"Training completed successfully for method {sub_job.extractor_job.method_name}"
            else:
                return False, "Training completed but model upload failed"
        elif sub_job.status == JobStatus.FAILURE:
            self.distributed_jobs.remove(distributed_job)
            return False, f"Training failed for method {sub_job.extractor_job.method_name}"
        else:
            return False, "Training in progress"

    def _process_performance_job(self, distributed_job: DistributedJob) -> Tuple[bool, str]:
        self.job_executor.update_job_statuses(distributed_job)

        for sub_job in distributed_job.sub_jobs:
            if sub_job.status == JobStatus.WAITING:
                self.job_executor.start_performance_evaluation(distributed_job.extraction_identifier, sub_job)

            if sub_job.result.is_perfect:
                break

        perfect_score_jobs = [job for job in distributed_job.sub_jobs if job.result and job.result.is_perfect]
        if perfect_score_jobs:
            self.job_executor.cancel_jobs(distributed_job)

        all_complete = all(sub_job.status in self.job_executor.get_finished_status() for sub_job in distributed_job.sub_jobs)

        if not all_complete:
            return False, "Performance evaluation in progress"

        self.distributed_jobs.remove(distributed_job)
        best_job: DistributedSubJob = JobSelectorUseCase.select_best_job(distributed_job)

        if not best_job:
            return False, "No valid performance results to select the best model"

        if not best_job.extractor_job.should_be_retrained_with_more_data:
            if self.job_executor.upload_model(distributed_job.extraction_identifier, best_job.extractor_job):
                return (
                    True,
                    f"Best model selected: {best_job.extractor_job.method_name} with performance {best_job.result.performance_score}",
                )
            else:
                return False, "Best model selected but upload failed"

        training_job = DistributedJob(
            extraction_identifier=distributed_job.extraction_identifier,
            type=JobType.TRAIN,
            sub_jobs=[DistributedSubJob(extractor_job=best_job.extractor_job)],
        )
        self.distributed_jobs.append(training_job)
        return False, "Retraining model"

    def exists_jobs_to_be_done(self) -> bool:
        return len(self.distributed_jobs) > 0
