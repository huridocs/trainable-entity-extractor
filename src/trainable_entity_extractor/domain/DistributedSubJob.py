from pydantic import BaseModel

from trainable_entity_extractor.domain.JobStatus import JobStatus
from trainable_entity_extractor.domain.Performance import Performance
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob


class DistributedSubJob(BaseModel):
    job_id: str | None = None
    extractor_job: TrainableEntityExtractorJob
    retry_count: int = 0
    max_retries: int = 2
    status: JobStatus = JobStatus.PENDING
    result: Performance | bool | None = None
