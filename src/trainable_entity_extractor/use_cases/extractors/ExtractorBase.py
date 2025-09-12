from abc import abstractmethod
import time
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.Performance import Performance
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.use_cases.send_logs import send_logs


class ExtractorBase:

    METHODS = list()

    def __init__(self, extraction_identifier: ExtractionIdentifier):
        self.extraction_identifier = extraction_identifier

    def get_name(self):
        return self.__class__.__name__

    @abstractmethod
    def create_model(self, extraction_data: ExtractionData) -> tuple[bool, str]:
        pass

    @abstractmethod
    def get_suggestions(self, predictions_samples: list[PredictionSample]) -> list[Suggestion]:
        pass

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
                method_instance = method

            if hasattr(method_instance, "can_be_used"):
                if not method_instance.can_be_used(extraction_data):
                    continue

            job = TrainableEntityExtractorJob(
                run_name=extraction_data.extraction_identifier.run_name,
                extraction_name=extraction_data.extraction_identifier.extraction_name,
                extractor_name=self.get_name(),
                method_name=method_instance.get_name(),
                gpu_needed=getattr(method_instance, "gpu_needed", False),
                timeout=getattr(method_instance, "timeout", 3600),
            )
            jobs.append(job)

        return jobs

    def get_performance(self, extractor_job: TrainableEntityExtractorJob, extraction_data: ExtractionData) -> Performance:
        method_name = extractor_job.method_name
        start_time = time.time()

        method_instance = self._get_method_instance_by_name(method_name)
        if not method_instance:
            send_logs(extraction_data.extraction_identifier, f"Method {method_name} not found")
            return Performance()

        if hasattr(method_instance, "can_be_used"):
            if not method_instance.can_be_used(extraction_data):
                send_logs(extraction_data.extraction_identifier, f"Method {method_name} cannot be used with current data")
                return Performance()

        send_logs(extraction_data.extraction_identifier, f"\nChecking {method_name}")

        try:
            train_set, test_set = self.prepare_for_training(extraction_data)
            performance_score = method_instance.get_performance(train_set, test_set)
            should_be_retrained_with_more_data = (
                method_instance.should_be_retrained_with_more_data()
                if hasattr(method_instance, "should_be_retrained_with_more_data")
                else True
            )
            performance_score = float(performance_score) if performance_score is not None else 0.0
        except Exception as e:
            should_be_retrained_with_more_data = True
            send_logs(extraction_data.extraction_identifier, "ERROR", LogSeverity.info, e)
            performance_score = 0.0

        execution_time = int(time.time() - start_time)
        return Performance(
            performance=performance_score,
            execution_seconds=execution_time,
            should_be_retrained_with_more_data=should_be_retrained_with_more_data,
        )

    def train_one_method(
        self, extractor_job: TrainableEntityExtractorJob, extraction_data: ExtractionData
    ) -> tuple[bool, str]:
        method_name = extractor_job.method_name
        start_time = time.time()

        method_instance = self._get_method_instance_by_name(method_name)
        if not method_instance:
            return False, f"Method {method_name} not found"

        if hasattr(method_instance, "can_be_used"):
            if not method_instance.can_be_used(extraction_data):
                return False, f"Method {method_name} cannot be used with current data"

        try:
            self.prepare_for_training(extraction_data)
            training_success = method_instance.train(extraction_data)

            if isinstance(training_success, tuple):
                success = training_success[0]
                message = training_success[1] if len(training_success) > 1 else ""
                if message:
                    send_logs(extraction_data.extraction_identifier, f"Training result: {message}")
            else:
                success = bool(training_success)
                message = ""

            execution_time = int(time.time() - start_time)
            status = "" if success else "failed"
            status_msg = f"Training {method_name} {status} in {execution_time}s"
            send_logs(extraction_data.extraction_identifier, status_msg)

            if success:
                return True, message if message else status_msg
            else:
                return False, message if message else f"Training {method_name} failed"

        except Exception as e:
            error_msg = f"Training {method_name} failed with error: {str(e)}"
            send_logs(extraction_data.extraction_identifier, error_msg, LogSeverity.error)
            send_logs(extraction_data.extraction_identifier, "ERROR", LogSeverity.info, e)
            return False, error_msg

    def _get_method_instance_by_name(self, method_name: str):
        for method in self.METHODS:
            if isinstance(method, type):
                method_instance = method(self.extraction_identifier)
            else:
                method_instance = method

            if method_instance.get_name() == method_name:
                return method_instance

        return None
