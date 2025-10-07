from abc import abstractmethod
import time
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.Performance import Performance
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.ports.MethodBase import MethodBase
from trainable_entity_extractor.ports.Logger import Logger


class ExtractorBase:

    METHODS: list[type[MethodBase]] = list()

    def __init__(self, extraction_identifier: ExtractionIdentifier, logger: Logger):
        self.extraction_identifier = extraction_identifier
        self.logger = logger

    def get_name(self):
        return self.__class__.__name__

    @abstractmethod
    def get_suggestions(self, method_name: str, prediction_samples: PredictionSamplesData) -> list[Suggestion]:
        pass

    def get_method_instance_by_name(self, method_name: str) -> MethodBase:
        for method in self.METHODS:
            if isinstance(method, type):
                method_instance = method(self.extraction_identifier)
            else:
                method_instance = method.set_extraction_identifier(self.extraction_identifier)

            if method_instance.get_name() == method_name:
                return method_instance

        raise ValueError(f"Method {method_name} not found in {self.get_name()}")

    @abstractmethod
    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        pass

    @abstractmethod
    def prepare_for_training(self, extraction_data: ExtractionData) -> tuple[ExtractionData, ExtractionData]:
        pass

    @staticmethod
    def is_multilingual(multi_option_data: ExtractionData) -> bool:
        not_multilingual_languages = ["", "en", "eng"]

        for sample in multi_option_data.samples:
            if sample.labeled_data.language_iso not in not_multilingual_languages:
                return True

        return False

    @staticmethod
    def get_train_test_sets(extraction_data: ExtractionData) -> tuple[ExtractionData, ExtractionData]:
        if len(extraction_data.samples) < 8:
            return extraction_data, extraction_data

        train_size = int(len(extraction_data.samples) * 0.8)

        train_set: list[TrainingSample] = extraction_data.samples[:train_size]

        if len(extraction_data.samples) < 15:
            test_set: list[TrainingSample] = extraction_data.samples[-10:]
        else:
            test_set = extraction_data.samples[train_size:]

        train_extraction_data = ExtractorBase.get_extraction_data_from_samples(extraction_data, train_set)
        test_extraction_data = ExtractorBase.get_extraction_data_from_samples(extraction_data, test_set)
        return train_extraction_data, test_extraction_data

    @staticmethod
    def get_extraction_data_from_samples(extraction_data: ExtractionData, samples: list[TrainingSample]) -> ExtractionData:
        return ExtractionData(
            samples=samples,
            options=extraction_data.options,
            multi_value=extraction_data.multi_value,
            extraction_identifier=extraction_data.extraction_identifier,
        )

    def get_distributed_jobs(self, extraction_data: ExtractionData) -> list[TrainableEntityExtractorJob]:
        jobs = list()
        for method in self.METHODS:
            if isinstance(method, type):
                method_instance = method(self.extraction_identifier)
            else:
                method_instance = method.set_extraction_identifier(self.extraction_identifier)

            if hasattr(method_instance, "can_be_used"):
                if not method_instance.can_be_used(extraction_data):
                    continue

            if hasattr(method_instance, "gpu_needed"):
                gpu_needed = method_instance.gpu_needed()
            else:
                gpu_needed = True

            job = TrainableEntityExtractorJob(
                run_name=extraction_data.extraction_identifier.run_name,
                extraction_name=extraction_data.extraction_identifier.extraction_name,
                extractor_name=self.get_name(),
                method_name=method_instance.get_name(),
                gpu_needed=gpu_needed,
                timeout=getattr(method_instance, "timeout", 3600),
                options=extraction_data.options if extraction_data.options else [],
                multi_value=extraction_data.multi_value if extraction_data.multi_value else False,
                metadata=(
                    extraction_data.extraction_identifier.metadata if extraction_data.extraction_identifier.metadata else {}
                ),
            )
            jobs.append(job)

        return jobs

    def get_performance(self, extractor_job: TrainableEntityExtractorJob, extraction_data: ExtractionData) -> Performance:
        method_name = extractor_job.method_name
        start_time = time.time()

        method_instance = self.get_method_instance_by_name(method_name)
        if not method_instance:
            self.logger.log(extraction_data.extraction_identifier, f"Method {method_name} not found")
            return Performance(method_name=method_name, failed=True)

        if hasattr(method_instance, "can_be_used"):
            if not method_instance.can_be_used(extraction_data):
                self.logger.log(
                    extraction_data.extraction_identifier, f"Method {method_name} cannot be used with current data"
                )
                return Performance(method_name=method_name, failed=True)

        self.logger.log(extraction_data.extraction_identifier, f"\nChecking {method_name}")

        try:
            train_set, test_set = self.prepare_for_training(extraction_data)
            performance_score = method_instance.get_performance(train_set, test_set)
            performance_score = float(performance_score) if performance_score is not None else 0.0

            execution_time = int(time.time() - start_time)
            is_perfect = performance_score >= 99.99

            return Performance(
                method_name=method_name,
                performance=performance_score,
                execution_seconds=execution_time,
                is_perfect=is_perfect,
                failed=False,
                testing_samples_count=len(test_set.samples),
                training_samples_count=len(train_set.samples),
                samples_count=len(extraction_data.samples),
            )

        except Exception as e:
            self.logger.log(extraction_data.extraction_identifier, "ERROR", LogSeverity.info, e)
            execution_time = int(time.time() - start_time)

            return Performance(method_name=method_name, execution_seconds=execution_time)

    def train_one_method(
        self, extractor_job: TrainableEntityExtractorJob, extraction_data: ExtractionData
    ) -> tuple[bool, str]:
        method_name = extractor_job.method_name
        method_instance = self.get_method_instance_by_name(method_name)
        if not method_instance:
            return False, f"Method {method_name} not found"

        if hasattr(method_instance, "can_be_used"):
            if not method_instance.can_be_used(extraction_data):
                return False, f"Method {method_name} cannot be used with current data"

        try:
            self.prepare_for_training(extraction_data)
            method_instance.train(extraction_data)
            return True, ""

        except Exception as e:
            error_msg = f"Training {method_name} failed with error: {str(e)}"
            self.logger.log(extraction_data.extraction_identifier, error_msg, LogSeverity.error)
            self.logger.log(extraction_data.extraction_identifier, "ERROR", LogSeverity.info, e)
            return False, error_msg
