from pydantic import BaseModel, Field
from datetime import datetime

from trainable_entity_extractor.domain.DistributedSubJob import DistributedSubJob
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.JobType import JobType


class DistributedJob(BaseModel):
    type: JobType
    sub_jobs: list[DistributedSubJob]
    start_time: datetime = Field(default_factory=datetime.now)
    domain_name: str = "default"
    extraction_identifier: ExtractionIdentifier
