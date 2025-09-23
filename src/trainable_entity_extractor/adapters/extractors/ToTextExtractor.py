from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.domain.PerformanceSummary import PerformanceSummary
from trainable_entity_extractor.ports.ExtractorBase import ExtractorBase
from trainable_entity_extractor.ports.Logger import Logger
from trainable_entity_extractor.adapters.extractors.ToTextExtractorMethod import ToTextExtractorMethod

RETRAIN_SAMPLES_THRESHOLD = 250


class ToTextExtractor(ExtractorBase):
    METHODS: list[type[ToTextExtractorMethod]] = []

    def __init__(self, extraction_identifier: ExtractionIdentifier, logger: Logger):
        super().__init__(extraction_identifier, logger)

    def prepare_for_training(self, extraction_data: ExtractionData) -> tuple[ExtractionData, ExtractionData]:
        """Base preparation for text extractors - return train/test sets"""
        return self.get_train_test_sets(extraction_data)

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        pass

    def get_name(self):
        return self.__class__.__name__

    def get_suggestions(self, method_name: str, predictions_samples: list[PredictionSample]) -> list[Suggestion]:
        method_instance = self.get_predictions_method(method_name)
        self.logger.log(
            self.extraction_identifier,
            f"And also using {method_instance.get_name()} to calculate {len(predictions_samples)} suggestions",
        )
        prediction = method_instance.predict(predictions_samples)
        suggestions = list()
        for prediction, prediction_sample in zip(prediction, predictions_samples):
            entity_name = prediction_sample.entity_name
            suggestions.append(Suggestion.from_prediction_text(self.extraction_identifier, entity_name, prediction))

        return suggestions

    def create_model(self, extraction_data: ExtractionData) -> tuple[bool, str]:
        samples_info = (
            f"\n{self.get_name()} received {len(extraction_data.samples)} samples from "
            f"{extraction_data.extraction_identifier.extraction_name}"
        )
        self.logger.log(self.extraction_identifier, samples_info)

        performance_summary = PerformanceSummary()

        perfect_score_method = None
        try:
            for method in self.METHODS:
                method_instance = method(self.extraction_identifier)

                if method_instance.can_be_used(extraction_data):
                    train_set, test_set = self.prepare_for_training(extraction_data)
                    performance_score = method_instance.get_performance(train_set, test_set)

                    performance_summary.add_performance(method_instance.get_name(), performance_score, test_set)

                    if performance_score >= 1:
                        perfect_score_method = method_instance
                        break

                self.logger.log(self.extraction_identifier, f"Checking {method_instance.get_name()}")

                try:
                    method_instance.train(extraction_data)
                except Exception as e:
                    message = f"{method_instance.get_name()} failed with error {e}"
                    self.logger.log(self.extraction_identifier, message, LogSeverity.error, e)
                    continue

                self.logger.log(self.extraction_identifier, performance_summary.to_log())

        except Exception as e:
            return False, f"ToTextExtractor.create_model error: {e}"

        self.logger.log(self.extraction_identifier, performance_summary.to_log())

        return True, "Model created successfully"
