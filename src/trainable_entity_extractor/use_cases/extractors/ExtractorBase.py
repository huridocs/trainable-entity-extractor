from abc import abstractmethod
import time
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionDistributedTask import ExtractionDistributedTask
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
    def prepare_for_performance(self, extraction_data: ExtractionData) -> tuple[ExtractionData, ExtractionData]:
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

    def get_distributed_tasks(self, extraction_data: ExtractionData) -> list[ExtractionDistributedTask]:
        tasks = list()
        for method in self.METHODS:
            if isinstance(method, type):
                method_instance = method(self.extraction_identifier)
            else:
                method_instance = method

            if hasattr(method_instance, "can_be_used"):
                if not method_instance.can_be_used(extraction_data):
                    continue

            task = ExtractionDistributedTask(
                run_name=extraction_data.extraction_identifier.run_name,
                extraction_name=extraction_data.extraction_identifier.extraction_name,
                extractor_name=self.get_name(),
                method_name=method_instance.get_name(),
                gpu_needed=getattr(method_instance, "gpu_needed", False),
                timeout=getattr(method_instance, "timeout", 3600),
            )
            tasks.append(task)

        return tasks

    def get_performance(
        self, extraction_distributed_task: ExtractionDistributedTask, extraction_data: ExtractionData
    ) -> Performance:
        method_name = extraction_distributed_task.method_name
        start_time = time.time()

        method_instance = self._get_method_instance_by_name(method_name)
        if not method_instance:
            send_logs(extraction_data.extraction_identifier, f"Method {method_name} not found")
            return Performance(method_name=method_name, performance=0.0, execution_seconds=0)

        if hasattr(method_instance, "can_be_used"):
            if not method_instance.can_be_used(extraction_data):
                send_logs(extraction_data.extraction_identifier, f"Method {method_name} cannot be used with current data")
                execution_time = int(time.time() - start_time)
                return Performance(method_name=method_name, performance=0.0, execution_seconds=execution_time)

        send_logs(extraction_data.extraction_identifier, f"\nChecking {method_name}")

        try:
            train_set, test_set = self.prepare_for_performance(extraction_data)
            performance_score = method_instance.get_performance(train_set, test_set)
            performance_score = float(performance_score) if performance_score is not None else 0.0
        except Exception as e:
            send_logs(extraction_data.extraction_identifier, "ERROR", LogSeverity.info, e)
            performance_score = 0.0

        execution_time = int(time.time() - start_time)
        return Performance(method_name=method_name, performance=performance_score, execution_seconds=execution_time)

    def _get_method_instance_by_name(self, method_name: str):
        for method in self.METHODS:
            if isinstance(method, type):
                method_instance = method(self.extraction_identifier)
            else:
                method_instance = method

            if method_instance.get_name() == method_name:
                return method_instance

        return None
