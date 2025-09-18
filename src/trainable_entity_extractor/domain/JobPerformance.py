from pydantic import BaseModel
from trainable_entity_extractor.domain.Performance import Performance
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob

from abc import ABC, abstractmethod


class JobPerformance(BaseModel):
    """Represents the performance result of a training job"""

    extractor_job: TrainableEntityExtractorJob
    performance_result: Performance
    job_id: str

    @property
    def performance_score(self) -> float:
        """Get the performance score"""
        return self.performance_result.performance

    @property
    def is_perfect(self) -> bool:
        """Check if the performance is perfect (100%)"""
        return self.performance_score == 100.0

    @property
    def method_name(self) -> str:
        """Get the method name from the extractor job"""
        return self.extractor_job.method_name
