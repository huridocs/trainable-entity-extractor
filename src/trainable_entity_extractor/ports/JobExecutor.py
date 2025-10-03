import shutil
from abc import ABC, abstractmethod

from trainable_entity_extractor.config import CACHE_PATH
from trainable_entity_extractor.domain.DistributedJob import DistributedJob
from trainable_entity_extractor.domain.DistributedSubJob import DistributedSubJob
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.JobStatus import JobStatus
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.ports.ExtractionDataRetriever import ExtractionDataRetriever
from trainable_entity_extractor.ports.ExtractorBase import ExtractorBase
from trainable_entity_extractor.ports.Logger import Logger
from trainable_entity_extractor.ports.ModelStorage import ModelStorage


class JobExecutor(ABC):
    def __init__(
        self,
        extractors: list[type[ExtractorBase]],
        data_retriever: ExtractionDataRetriever,
        model_storage: ModelStorage,
        logger: Logger,
    ):
        self.extractors = extractors
        self.data_retriever = data_retriever
        self.model_storage = model_storage
        self.logger = logger

    @abstractmethod
    def start_performance_evaluation(
        self, extraction_identifier: ExtractionIdentifier, distributed_sub_job: DistributedSubJob
    ):
        pass

    @abstractmethod
    def start_training(self, extraction_identifier: ExtractionIdentifier, distributed_sub_job: DistributedSubJob):
        pass

    @abstractmethod
    def start_prediction(self, extraction_identifier: ExtractionIdentifier, distributed_sub_job: DistributedSubJob):
        pass

    @abstractmethod
    def update_job_statuses(self, distributed_job: DistributedJob):
        pass

    @abstractmethod
    def cancel_jobs(self, job: DistributedJob) -> None:
        pass

    @staticmethod
    def get_finished_status() -> list[JobStatus]:
        return [JobStatus.SUCCESS, JobStatus.FAILURE, JobStatus.CANCELED]

    @staticmethod
    def recreate_model_folder(extraction_identifier: ExtractionIdentifier) -> None:
        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)
        extraction_identifier.get_path().mkdir(parents=True, exist_ok=True)

    def is_extractor_cancelled(self, extractor_identifier: ExtractionIdentifier) -> bool:
        try:
            return self.data_retriever.is_extractor_cancelled(extractor_identifier)
        except Exception as e:
            self.logger.log(
                extractor_identifier,
                f"Error checking if extractor is cancelled: {e}",
                LogSeverity.error,
            )
            return False
