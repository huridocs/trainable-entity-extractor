import os
from collections import Counter
from typing import Type

from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.domain.Value import Value

from trainable_entity_extractor.use_cases.extractors.ExtractorBase import ExtractorBase
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import (
    TextToMultiOptionMethod,
)
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.FirstWordRegex import (
    FirstWordRegex,
)
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.TextBalancedSetFit import (
    TextBalancedSetFit,
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
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.TextSetFitMultilingual import (
    TextSetFitMultilingual,
)
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.TextSingleLabelSetFit import (
    TextSingleLabelSetFit,
)
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.TextSingleLabelSetFitMultilingual import (
    TextSingleLabelSetFitMultilingual,
)
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.TextToCountries import (
    TextToCountries,
)
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.gemini_multi_option.TextGeminiMultiOption import (
    TextGeminiMultiOption,
)
from trainable_entity_extractor.use_cases.send_logs import send_logs


class TextToMultiOptionExtractor(ExtractorBase):
    SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

    METHODS: list[Type[TextToMultiOptionMethod]] = [
        NaiveTextToMultiOptionMethod,
        TextToCountries,
        FirstWordRegex,
        TextFuzzyFirst,
        TextFuzzyFirstCleanLabels,
        TextFuzzyLast,
        TextFuzzyLastCleanLabels,
        TextFuzzyAll100,
        TextFuzzyAll88,
        TextFuzzyAll75,
        TextGeminiMultiOption,
        TextBalancedSetFit,
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
            segment_text = prediction_sample.source_text if prediction_sample.source_text else ""
            values = [Value(id=x.id, label=x.label, segment_text=segment_text) for x in prediction]
            suggestion = Suggestion.from_prediction_multi_option(
                self.extraction_identifier, prediction_sample.entity_name, values
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

        send_logs(self.extraction_identifier, self.get_stats(extraction_data))
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
        except IndexError:
            performance = 0
            if "setfit" in method_instance.get_name().lower():
                config_logger.info("Insufficient data to train SetFit model")
            else:
                config_logger.info("ERROR", exc_info=True)
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

    def get_train_test_sets(self, extraction_data: ExtractionData) -> (ExtractionData, ExtractionData):
        if len(extraction_data.samples) < 8:
            return extraction_data, extraction_data

        samples_by_labels = self.get_samples_by_labels(extraction_data)

        if self.percentage_of_labels_without_values(samples_by_labels) > 15:
            train_set, test_set = self.get_train_test_sets_using_labels(samples_by_labels)
        else:
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
    def get_samples_by_labels(extraction_data):
        samples_by_labels = {option.label: [] for option in extraction_data.options}
        for sample in extraction_data.samples:
            sample_labels = [option.label for option in sample.labeled_data.values]
            for label in sample_labels:
                if label in samples_by_labels:
                    samples_by_labels[label].append(sample)

        return samples_by_labels

    @staticmethod
    def percentage_of_labels_without_values(samples_by_labels: dict[str, list[TrainingSample]]) -> float:
        total_samples = sum(len(samples) for samples in samples_by_labels.values())
        if total_samples == 0:
            return 0.0

        samples_without_values = sum(1 for samples in samples_by_labels.values() if not samples)
        return (samples_without_values / total_samples) * 100

    @staticmethod
    def get_train_test_sets_using_labels(
        samples_by_labels: dict[str, list[TrainingSample]]
    ) -> (list[TrainingSample], list[TrainingSample]):
        test_set = set()
        all_samples = {sample for samples in samples_by_labels.values() for sample in samples}
        sorted_labels_by_samples_count = sorted(samples_by_labels.keys(), key=lambda x: len(samples_by_labels[x]))

        for label in sorted_labels_by_samples_count:
            test_set.update(samples_by_labels[label])
            if len(test_set) / len(all_samples) >= 0.10:
                break
        if len(all_samples) - len(test_set) < 8:
            test_set = set(list(all_samples)[: int(len(all_samples) * 0.30)])
        else:
            test_size = int(len(all_samples) * 0.10)
            test_set.update(list(all_samples)[:test_size])

        train_set = all_samples - test_set
        return list(train_set), list(test_set)

    @staticmethod
    def get_stats(extraction_data):
        options_with_samples = Counter()
        for sample in extraction_data.samples:
            options_with_samples.update([option.label for option in sample.labeled_data.values])

        languages = Counter()
        for sample in extraction_data.samples:
            languages.update([sample.labeled_data.language_iso])

        all_option_labels = {opt.label for opt in extraction_data.options}
        options_with_no_samples = all_option_labels - set(options_with_samples.keys())

        options_count = len(extraction_data.options)
        stats = f"\nNumber of options: {options_count}\n"
        stats += f"Number of samples: {len(extraction_data.samples)}\n"

        stats += f"Languages\n"
        stats += "\n".join([f"{key} {value}" for key, value in languages.most_common()])
        stats += f"\nOptions with samples\n"
        stats += "\n".join([f"{key} {value}" for key, value in options_with_samples.most_common()])

        if options_with_no_samples:
            stats += f"\nOptions with no samples\n"
            stats += "\n".join(list(options_with_no_samples))

        return stats
