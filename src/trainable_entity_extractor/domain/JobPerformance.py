from pydantic import BaseModel
from trainable_entity_extractor.domain.DistributedPerformance import DistributedPerformance
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob


class JobPerformance(BaseModel):

    extractor_job: TrainableEntityExtractorJob
    performance_result: DistributedPerformance
    job_id: str

    @property
    def performance_score(self) -> float:
        return self.performance_result.performance

    @property
    def is_perfect(self) -> bool:
        return self.performance_score == 100.0

    @property
    def method_name(self) -> str:
        return self.extractor_job.method_name

    @property
    def should_be_retrained_with_more_data(self) -> bool:
        return self.extractor_job.should_be_retrained_with_more_data
