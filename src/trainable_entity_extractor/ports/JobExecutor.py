from abc import ABC, abstractmethod
from typing import Tuple
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.Performance import Performance


class JobExecutor(ABC):

    @abstractmethod
    def execute_training(
        self, extractor_job: TrainableEntityExtractorJob, options: list, multi_value: bool
    ) -> Tuple[bool, str]:
        pass

    @abstractmethod
    def execute_prediction(self, extractor_job: TrainableEntityExtractorJob) -> Tuple[bool, str]:
        pass

    @abstractmethod
    def execute_performance_evaluation(
        self, extractor_job: TrainableEntityExtractorJob, options: list, multi_value: bool
    ) -> Performance:
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> str:
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> None:
        pass
