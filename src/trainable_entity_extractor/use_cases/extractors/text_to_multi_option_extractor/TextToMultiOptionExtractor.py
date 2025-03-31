import os
from typing import Type

from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.use_cases.extractors.ExtractorBase import ExtractorBase
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import (
    TextToMultiOptionMethod,
)
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.FirstWordRegex import (
    FirstWordRegex,
)
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.TextFastTextMethod import (
    TextFastTextMethod,
)
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.TextFuzzyAll100 import (
    TextFuzzyAll100,
)
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.TextFuzzyAll75 import (
    TextFuzzyAll75,
)
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.TextFuzzyAll88 import (
    TextFuzzyAll88,
)
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.TextFuzzyFirst import (
    TextFuzzyFirst,
)
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.NaiveTextToMultiOptionMethod import (
    NaiveTextToMultiOptionMethod,
)
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.TextFuzzyFirstCleanLabels import (
    TextFuzzyFirstCleanLabels,
)
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.TextFuzzyLast import (
    TextFuzzyLast,
)
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.TextFuzzyLastCleanLabels import (
    TextFuzzyLastCleanLabels,
)
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.TextSetFit import TextSetFit
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.TextSetFitMultilingual import (
    TextSetFitMultilingual,
)
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.TextSingleLabelSetFit import (
    TextSingleLabelSetFit,
)
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.TextSingleLabelSetFitMultilingual import (
    TextSingleLabelSetFitMultilingual,
)
from trainable_entity_extractor.use_cases.send_logs import send_logs


class TextToMultiOptionExtractor(ExtractorBase):
    SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

    METHODS: list[Type[TextToMultiOptionMethod]] = [
        NaiveTextToMultiOptionMethod,
        FirstWordRegex,
        TextFuzzyFirst,
        TextFuzzyFirstCleanLabels,
        TextFuzzyLast,
        TextFuzzyLastCleanLabels,
        TextFuzzyAll100,
        TextFuzzyAll88,
        TextFuzzyAll75,
        TextFastTextMethod,
        TextSetFit,
        TextSetFitMultilingual,
        TextSingleLabelSetFit,
        TextSingleLabelSetFitMultilingual,
    ]

    def __init__(self, extraction_identifier: ExtractionIdentifier):
        super().__init__(extraction_identifier)

        self.options: list[Option] = list()
        self.multi_value = False

    def get_suggestions(self, predictions_samples: list[PredictionSample]) -> list[Suggestion]:
        if not predictions_samples:
            return []

        predictions = self.get_predictions_method().predict(predictions_samples)

        if not self.multi_value:
            predictions = [x[:1] for x in predictions]

        suggestions = list()
        for prediction_sample, prediction in zip(predictions_samples, predictions):
            suggestion = Suggestion.from_prediction_multi_option(
                self.extraction_identifier, prediction_sample.entity_name, prediction
            )
            suggestions.append(suggestion)

        return suggestions

    def get_predictions_method(self):
        self.options = self.extraction_identifier.get_options()
        self.multi_value = self.extraction_identifier.get_multi_value()
        method_name = self.extraction_identifier.get_method_used()
        for method in self.METHODS:
            method_instance = method(self.extraction_identifier, self.options, self.multi_value)
            if method_instance.get_name() == method_name:
                return method_instance

        return self.METHODS[0](self.extraction_identifier, self.options, self.multi_value)

    def create_model(self, extraction_data: ExtractionData) -> tuple[bool, str]:
        self.extraction_identifier.save_options(extraction_data.options)
        self.options = extraction_data.options
        self.multi_value = extraction_data.multi_value

        best_method_instance = self.get_best_method(extraction_data)
        best_method_instance.train(extraction_data)

        self.extraction_identifier.save_multi_value(extraction_data.multi_value)
        self.extraction_identifier.save_method_used(best_method_instance.get_name())
        return True, ""

    def get_best_method(self, extraction_data: ExtractionData):
        best_performance = 0
        best_method_instance = self.METHODS[0](self.extraction_identifier, self.options, self.multi_value)
        performance_log = "Performance aggregation:\n"

        for method in self.METHODS:
            method_instance = method(self.extraction_identifier, self.options, self.multi_value)

            if len(self.METHODS) == 1:
                return method_instance

            if not method_instance.can_be_used(extraction_data):
                continue

            performance = self.get_performance(extraction_data, method_instance)
            performance_log += f"{method_instance.get_name()}: {round(performance, 2)}%\n"

            if performance == 100:
                send_logs(self.extraction_identifier, performance_log)
                send_logs(self.extraction_identifier, f"Best method {method_instance.get_name()} with {performance}%")
                self.extraction_identifier.save_content("performance_log.txt", performance_log)
                return method_instance

            if performance > best_performance:
                best_performance = performance
                best_method_instance = method_instance

        send_logs(self.extraction_identifier, performance_log)
        send_logs(self.extraction_identifier, f"Best method {best_method_instance.get_name()} with {best_performance}%")
        self.extraction_identifier.save_content("performance_log.txt", performance_log)
        return best_method_instance

    @staticmethod
    def get_performance(extraction_data, method_instance):
        config_logger.info(f"\nChecking {method_instance.get_name()}")
        try:
            performance = method_instance.performance(extraction_data)
        except:
            config_logger.info("ERROR", exc_info=True)
            performance = 0
        config_logger.info(f"\nPerformance {method_instance.get_name()}: {performance}%")
        return performance

    def remove_models(self):
        for method in self.METHODS:
            method_instance = method(self.extraction_identifier, self.options, self.multi_value)
            method_instance.remove_model()

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        if not extraction_data.options and not extraction_data.extraction_identifier.get_options_path().exists():
            return False

        for sample in extraction_data.samples:
            if sample.labeled_data or sample.labeled_data.source_text:
                return True

        return False

    def suggestions_from_predictions(
        self, method_instance: type[TextToMultiOptionMethod], predictions_samples: list[PredictionSample]
    ) -> list[Suggestion]:
        suggestions = list()
        prediction = method_instance.predict(predictions_samples, self.options)

        for prediction, prediction_sample in zip(prediction, predictions_samples):
            suggestion = Suggestion.from_prediction_multi_option(
                self.extraction_identifier, prediction_sample.entity_name, prediction
            )
            suggestions.append(suggestion)

        return suggestions
