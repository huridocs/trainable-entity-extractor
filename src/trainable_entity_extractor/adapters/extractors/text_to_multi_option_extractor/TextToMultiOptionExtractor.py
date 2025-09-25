import os
from collections import Counter
from typing import Type

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.domain.PerformanceSummary import PerformanceSummary

from trainable_entity_extractor.ports.ExtractorBase import ExtractorBase
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import (
    TextToMultiOptionMethod,
)
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.methods.FirstWordRegex import (
    FirstWordRegex,
)
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.methods.TextBalancedSetFit import (
    TextBalancedSetFit,
)
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.methods.TextFuzzyAll100 import (
    TextFuzzyAll100,
)
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.methods.TextFuzzyAll75 import (
    TextFuzzyAll75,
)
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.methods.TextFuzzyAll88 import (
    TextFuzzyAll88,
)
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.methods.TextFuzzyFirst import (
    TextFuzzyFirst,
)
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.methods.NaiveTextToMultiOptionMethod import (
    NaiveTextToMultiOptionMethod,
)
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.methods.TextFuzzyFirstCleanLabels import (
    TextFuzzyFirstCleanLabels,
)
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.methods.TextFuzzyLast import (
    TextFuzzyLast,
)
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.methods.TextFuzzyLastCleanLabels import (
    TextFuzzyLastCleanLabels,
)
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.methods.TextSetFitMultilingual import (
    TextSetFitMultilingual,
)
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.methods.TextSingleLabelSetFit import (
    TextSingleLabelSetFit,
)
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.methods.TextSingleLabelSetFitMultilingual import (
    TextSingleLabelSetFitMultilingual,
)
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.methods.TextToCountries import (
    TextToCountries,
)
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.methods.gemini_multi_option.TextGeminiMultiOption import (
    TextGeminiMultiOption,
)
from trainable_entity_extractor.ports.Logger import Logger


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
    EMPTY_PLACEHOLDER = "EMPTY"

    def __init__(self, extraction_identifier: ExtractionIdentifier, logger: Logger):
        super().__init__(extraction_identifier, logger)
        self.options: list[Option] = list()
        self.multi_value = False

    def prepare_for_training(self, extraction_data: ExtractionData) -> tuple[ExtractionData, ExtractionData]:
        self.fix_empty_data(extraction_data)
        self.options = extraction_data.options
        self.multi_value = extraction_data.multi_value
        return self.get_train_test_sets(extraction_data)

    def get_suggestions(self, method_name: str, prediction_samples_data: PredictionSamplesData) -> list[Suggestion]:
        if not prediction_samples_data.prediction_samples:
            return []

        self.fix_empty_prediction_data(prediction_samples_data.prediction_samples)
        predictions: list[list[Value]] = self.get_method_instance_by_name(method_name).predict(prediction_samples_data)

        if not prediction_samples_data.multi_value:
            predictions = [x[:1] for x in predictions]

        suggestions = list()
        for prediction_sample, prediction in zip(prediction_samples_data.prediction_samples, predictions):
            segment_text = prediction_sample.source_text if prediction_sample.source_text else ""
            values = [Value(id=x.id, label=x.label, segment_text=segment_text) for x in prediction]
            suggestion = Suggestion.from_prediction_multi_option(
                self.extraction_identifier, prediction_sample.entity_name, values
            )
            suggestions.append(suggestion)

        return suggestions

    def create_model(self, extraction_data: ExtractionData) -> tuple[bool, str]:
        self.logger.log(self.extraction_identifier, self.get_stats(extraction_data))

        performance_summary = PerformanceSummary()

        perfect_score_method = None
        try:
            for method in self.METHODS:
                method_instance = method(self.extraction_identifier)

                if not method_instance.can_be_used(extraction_data):
                    continue

                train_set, test_set = self.prepare_for_training(extraction_data)
                performance_score = method_instance.get_performance(train_set, test_set)

                performance_summary.add_performance(method_instance.get_name(), performance_score, test_set)

                if performance_score >= 1:
                    perfect_score_method = method_instance
                    break

                try:
                    method_instance.train(extraction_data)
                except Exception as e:
                    if "Insufficient data to train SetFit model" in str(e):
                        continue

            if perfect_score_method:
                self.logger.log(self.extraction_identifier, performance_summary.to_log())
                return True, "Perfect score method found"

        except Exception as e:
            return False, f"TextToMultiOptionExtractor.create_model error: {e}"

        self.logger.log(self.extraction_identifier, performance_summary.to_log())

        return True, "Model created successfully"

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        if not extraction_data.options and not extraction_data.extraction_identifier.get_options_path().exists():
            return False

        for sample in extraction_data.samples:
            if sample.labeled_data or sample.labeled_data.source_text:
                return True

        return False

    def get_train_test_sets(self, extraction_data: ExtractionData) -> tuple[ExtractionData, ExtractionData]:
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
    ) -> tuple[list[TrainingSample], list[TrainingSample]]:
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

    def fix_empty_data(self, extraction_data: ExtractionData):
        for sample in extraction_data.samples:
            if not sample.labeled_data.source_text.strip():
                sample.labeled_data.source_text = self.EMPTY_PLACEHOLDER

    def fix_empty_prediction_data(self, predictions_samples: list[PredictionSample]):
        for sample in predictions_samples:
            if not sample.source_text.strip():
                sample.source_text = self.EMPTY_PLACEHOLDER
