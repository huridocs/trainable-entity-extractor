from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.domain.PerformanceSummary import PerformanceSummary
from trainable_entity_extractor.use_cases.extractors.ExtractorBase import ExtractorBase
from trainable_entity_extractor.use_cases.extractors.ToTextExtractorMethod import ToTextExtractorMethod
from trainable_entity_extractor.use_cases.send_logs import send_logs

RETRAIN_SAMPLES_THRESHOLD = 250


class ToTextExtractor(ExtractorBase):
    METHODS: list[type[ToTextExtractorMethod]] = []

    def __init__(self, extraction_identifier: ExtractionIdentifier):
        super().__init__(extraction_identifier)

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        pass

    def get_name(self):
        return self.__class__.__name__

    def get_suggestions(self, predictions_samples: list[PredictionSample]) -> list[Suggestion]:
        method_instance = self.get_predictions_method()
        send_logs(
            self.extraction_identifier,
            f"And also using {method_instance.get_name()} to calculate {len(predictions_samples)} suggestions",
        )
        prediction = method_instance.predict(predictions_samples)
        suggestions = list()
        for prediction, prediction_sample in zip(prediction, predictions_samples):
            entity_name = prediction_sample.entity_name
            suggestions.append(Suggestion.from_prediction_text(self.extraction_identifier, entity_name, prediction))

        for suggestion, sample in zip(suggestions, predictions_samples):
            if sample.source_text:
                suggestion.segment_text = sample.source_text
            elif sample.pdf_data and sample.pdf_data.pdf_data_segments:
                suggestion.add_segments(sample.pdf_data)
            else:
                suggestion.segment_text = ""

        return suggestions

    def get_predictions_method(self) -> ToTextExtractorMethod:
        method_name = self.extraction_identifier.get_method_used()
        for method in self.METHODS:
            method_instance = method(self.extraction_identifier)
            if method_instance.get_name() == method_name:
                return method_instance

        return self.METHODS[0](self.extraction_identifier)

    def create_model(self, extraction_data: ExtractionData) -> tuple[bool, str]:
        if not extraction_data or not extraction_data.samples:
            return False, "No samples to create model"

        performance_train_set, performance_test_set = self.get_train_test_sets(extraction_data)

        samples_info = f"Train: {len(performance_train_set.samples)} samples\n"
        samples_info += f"Test: {len(performance_test_set.samples)} samples"
        send_logs(self.extraction_identifier, samples_info)

        if len(extraction_data.samples) < 2:
            best_method_instance = self.METHODS[0](self.extraction_identifier)
            config_logger.info(f"\nBest method {best_method_instance.get_name()} because no samples")
            best_method_instance.train(extraction_data)
            return True, ""

        best_method_instance = self.get_best_method(extraction_data)
        self.extraction_identifier.save_method_used(best_method_instance.get_name())

        if len(extraction_data.samples) < RETRAIN_SAMPLES_THRESHOLD:
            best_method_instance.train(extraction_data)

        # self.remove_data_from_methods_not_selected(best_method_instance)

        return True, ""

    @staticmethod
    def get_train_test_sets(extraction_data: ExtractionData) -> (ExtractionData, ExtractionData):
        return ExtractorBase.get_train_test_sets(extraction_data)

    def remove_data_from_methods_not_selected(self, best_method_instance):
        for method_to_remove in self.METHODS:
            method_instance = method_to_remove(self.extraction_identifier)
            if method_instance.get_name() != best_method_instance.get_name():
                method_instance.remove_method_data()

    def get_best_method(self, extraction_data: ExtractionData):
        best_performance = 0
        best_method_instance = self.METHODS[0](self.extraction_identifier)

        training_set, test_set = self.get_train_test_sets(extraction_data)
        performance_summary = PerformanceSummary.from_extraction_data(
            extractor_name=self.get_name(),
            training_samples_count=len(training_set.samples),
            testing_samples_count=len(test_set.samples),
            extraction_data=extraction_data,
        )

        for method in self.METHODS:
            if self.is_training_canceled():
                send_logs(self.extraction_identifier, "Training cancelled", LogSeverity.info)
                return best_method_instance

            method_instance = method(self.extraction_identifier)
            send_logs(self.extraction_identifier, f"Checking {method_instance.get_name()}")
            try:
                performance = method_instance.performance(training_set, test_set)
            except Exception as e:
                message = f"Error checking {method_instance.get_name()}"
                send_logs(self.extraction_identifier, message, LogSeverity.error, e)
                performance = 0
            performance_summary.add_performance(method_instance.get_name(), performance)
            if performance == 100:
                send_logs(self.extraction_identifier, performance_summary.to_log())
                self.extraction_identifier.save_content("performance_log.txt", performance_summary.to_log())
                return method_instance
            if performance > best_performance:
                best_performance = performance
                best_method_instance = method_instance

        send_logs(self.extraction_identifier, performance_summary.to_log())
        self.extraction_identifier.save_content("performance_log.txt", performance_summary.to_log())
        return best_method_instance
