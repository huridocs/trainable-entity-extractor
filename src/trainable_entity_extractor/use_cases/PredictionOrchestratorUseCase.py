from typing import Tuple, List
from trainable_entity_extractor.domain.DistributedJob import DistributedJob
from trainable_entity_extractor.domain.JobStatus import JobStatus
from trainable_entity_extractor.domain.JobType import JobType
from trainable_entity_extractor.ports.JobExecutor import JobExecutor
from trainable_entity_extractor.ports.Logger import Logger


class PredictionOrchestratorUseCase:
    def __init__(self, job_executor: JobExecutor, logger: Logger):
        self.job_executor = job_executor
        self.logger = logger
        self.distributed_jobs: List[DistributedJob] = []

    def process_job(self, distributed_job: DistributedJob) -> Tuple[bool, str]:
        if distributed_job.type == JobType.PREDICT:
            return self._process_prediction_job(distributed_job)
        else:
            self.distributed_jobs.remove(distributed_job)
            return False, f"Unknown job type: {distributed_job.type}"

    def _process_prediction_job(self, distributed_job: DistributedJob) -> Tuple[bool, str]:
        extraction_identifier = distributed_job.extraction_identifier
        sub_job = distributed_job.sub_jobs[0]

        if sub_job.status == JobStatus.WAITING:
            self.job_executor.start_prediction(extraction_identifier, sub_job)

        if sub_job.status == JobStatus.SUCCESS:
            self.distributed_jobs.remove(distributed_job)
            if sub_job.result:
                return True, f"Prediction completed successfully for method {sub_job.extractor_job.method_name}"
            else:
                return False, "Prediction completed but no results generated"
        elif sub_job.status == JobStatus.FAILURE:
            self.distributed_jobs.remove(distributed_job)
            return False, f"Prediction failed for method {sub_job.extractor_job.method_name}"
        else:
            return False, "Prediction in progress"

    def exists_jobs_to_be_done(self) -> bool:
        return len(self.distributed_jobs) > 0
