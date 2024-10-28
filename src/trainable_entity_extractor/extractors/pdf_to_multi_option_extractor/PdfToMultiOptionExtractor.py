import json
from collections import Counter
from os.path import join, exists
from pathlib import Path

from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.LogsMessage import Severity
from trainable_entity_extractor.data.Option import Option
from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.data.Suggestion import Suggestion
from trainable_entity_extractor.extractors.ExtractorBase import ExtractorBase
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.PdfMultiOptionMethod import PdfMultiOptionMethod

from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.TrainingSample import TrainingSample
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanBeginningDot1000 import (
    CleanBeginningDot1000,
)
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanBeginningDotDigits500 import (
    CleanBeginningDotDigits500,
)
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanEndDotDigits1000 import (
    CleanEndDotDigits1000,
)
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_labels_methods.FastTextMethod import (
    FastTextMethod,
)
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_labels_methods.SetFitEnglishMethod import (
    SetFitEnglishMethod,
)
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_labels_methods.SetFitMultilingualMethod import (
    SetFitMultilingualMethod,
)
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_labels_methods.SingleLabelSetFitEnglishMethod import (
    SingleLabelSetFitEnglishMethod,
)
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_labels_methods.SingleLabelSetFitMultilingualMethod import (
    SingleLabelSetFitMultilingualMethod,
)
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FastSegmentSelectorFuzzy95 import (
    FastSegmentSelectorFuzzy95,
)
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FastSegmentSelectorFuzzyCommas import (
    FastSegmentSelectorFuzzyCommas,
)
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyAll100 import (
    FuzzyAll100,
)
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyAll75 import (
    FuzzyAll75,
)
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyAll88 import (
    FuzzyAll88,
)
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyFirst import (
    FuzzyFirst,
)
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyFirstCleanLabel import (
    FuzzyFirstCleanLabel,
)
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyLast import (
    FuzzyLast,
)
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyLastCleanLabel import (
    FuzzyLastCleanLabel,
)
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzySegmentSelector import (
    FuzzySegmentSelector,
)
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.NextWordsTokenSelectorFuzzy75 import (
    NextWordsTokenSelectorFuzzy75,
)

from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.PreviousWordsSentenceSelectorFuzzyCommas import (
    PreviousWordsSentenceSelectorFuzzyCommas,
)
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.PreviousWordsTokenSelectorFuzzy75 import (
    PreviousWordsTokenSelectorFuzzy75,
)
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.SentenceSelectorFuzzyCommas import (
    SentenceSelectorFuzzyCommas,
)
from trainable_entity_extractor.extractors.segment_selector.FastAndPositionsSegmentSelector import (
    FastAndPositionsSegmentSelector,
)
from trainable_entity_extractor.extractors.segment_selector.FastSegmentSelector import FastSegmentSelector
from trainable_entity_extractor.extractors.segment_selector.SegmentSelector import SegmentSelector
from trainable_entity_extractor.send_logs import send_logs

RETRAIN_SAMPLES_THRESHOLD = 250


class PdfToMultiOptionExtractor(ExtractorBase):
    METHODS: list[PdfMultiOptionMethod] = [
        FuzzyFirst(),
        FuzzyLast(),
        FuzzyFirstCleanLabel(),
        FuzzyLastCleanLabel(),
        FuzzyAll75(),
        FuzzyAll88(),
        FuzzyAll100(),
        PreviousWordsTokenSelectorFuzzy75(),
        NextWordsTokenSelectorFuzzy75(),
        PreviousWordsSentenceSelectorFuzzyCommas(),
        SentenceSelectorFuzzyCommas(),
        FastSegmentSelectorFuzzy95(),
        FastSegmentSelectorFuzzyCommas(),
        FuzzySegmentSelector(),
        PdfMultiOptionMethod(CleanBeginningDotDigits500, FastTextMethod),
        PdfMultiOptionMethod(CleanEndDotDigits1000, FastTextMethod),
        PdfMultiOptionMethod(CleanBeginningDot1000, SetFitEnglishMethod),
        PdfMultiOptionMethod(CleanBeginningDot1000, SetFitMultilingualMethod),
        PdfMultiOptionMethod(CleanBeginningDot1000, SingleLabelSetFitEnglishMethod),
        PdfMultiOptionMethod(CleanBeginningDot1000, SingleLabelSetFitMultilingualMethod),
    ]

    def __init__(self, extraction_identifier: ExtractionIdentifier):
        super().__init__(extraction_identifier)
        self.base_path = join(self.extraction_identifier.get_path(), "multi_option_extractor")
        self.multi_value_path = join(self.base_path, "multi_value.json")
        self.method_name_path = Path(join(self.base_path, "method_name.json"))

        self.options: list[Option] = list()
        self.multi_value = False

    def create_model(self, extraction_data: ExtractionData):
        self.options = self.load_options(extraction_data.options)
        self.multi_value = extraction_data.multi_value
        send_logs(self.extraction_identifier, f"options {[x.model_dump() for x in self.options]}")

        SegmentSelector(self.extraction_identifier).prepare_model_folder()
        FastSegmentSelector(self.extraction_identifier).prepare_model_folder()
        FastAndPositionsSegmentSelector(self.extraction_identifier).prepare_model_folder()

        send_logs(self.extraction_identifier, self.get_stats(extraction_data))

        performance_train_set, performance_test_set = ExtractorBase.get_train_test_sets(extraction_data)
        samples_info = f"Train: {len(performance_train_set.samples)} samples\n"
        samples_info += f"Test: {len(performance_test_set.samples)} samples"
        send_logs(self.extraction_identifier, samples_info)

        method = self.get_best_method(extraction_data)

        for method_to_remove in [x for x in self.METHODS if x.get_name() != method.get_name()]:
            method_to_remove.remove_method_data(extraction_data.extraction_identifier)

        if len(extraction_data.samples) < RETRAIN_SAMPLES_THRESHOLD:
            method.train(extraction_data)

        self.save_json(self.multi_value_path, extraction_data.multi_value)
        self.save_json(str(self.method_name_path), method.get_name())

        return True, ""

    def get_suggestions(self, predictions_samples: list[PredictionSample]) -> list[Suggestion]:
        if not predictions_samples:
            return []

        training_samples, predictions = self.get_predictions(predictions_samples)
        prediction_method = self.get_predictions_method()

        context_from_the_end = "End" in prediction_method.get_name()
        suggestions = list()
        for training_sample, prediction_sample, prediction in zip(training_samples, predictions_samples, predictions):
            suggestion = Suggestion.get_empty(self.extraction_identifier, prediction_sample.entity_name)
            suggestion.add_prediction_multi_option(training_sample, prediction, context_from_the_end)
            suggestions.append(suggestion)

        return suggestions

    def get_predictions(self, predictions_samples: list[PredictionSample]) -> (list[TrainingSample], list[list[Option]]):
        self.options = self.load_options()
        self.multi_value = self.load_multi_value()
        training_samples = [TrainingSample(pdf_data=sample.pdf_data) for sample in predictions_samples]
        extraction_data = ExtractionData(
            multi_value=self.multi_value,
            options=self.options,
            samples=training_samples,
            extraction_identifier=self.extraction_identifier,
        )
        method = self.get_predictions_method()
        method.set_parameters(extraction_data)
        send_logs(self.extraction_identifier, f"Using method {method.get_name()} for suggestions")

        prediction = method.predict(extraction_data)

        if not self.multi_value:
            prediction = [x[:1] for x in prediction]

        return method.get_samples_for_context(extraction_data), prediction

    def load_options(self, options: list[Option] = None) -> list[Option]:
        if options:
            self.extraction_identifier.get_options_path().write_text(json.dumps([x.model_dump() for x in options]))
            return options

        if not exists(self.extraction_identifier.get_options_path()):
            return []

        return [Option(**x) for x in json.loads(self.extraction_identifier.get_options_path().read_text())]

    def load_multi_value(self) -> bool:
        if not exists(self.multi_value_path):
            return False

        with open(self.multi_value_path, "r") as file:
            return json.load(file)

    def get_best_method(self, multi_option_data: ExtractionData) -> PdfMultiOptionMethod:
        best_method_instance = self.METHODS[0]
        best_performance = 0
        performance_log = "Performance aggregation:\n"
        train_set, test_set = ExtractorBase.get_train_test_sets(multi_option_data)
        for method in self.METHODS:
            performance = self.get_method_performance(method, train_set, test_set)
            performance_log += f"{method.get_name()}: {round(performance, 2)}%\n"
            if performance == 100:
                send_logs(self.extraction_identifier, performance_log)
                send_logs(self.extraction_identifier, f"Best method {method.get_name()} with {performance}%")
                return method

            if performance > best_performance:
                best_performance = performance
                best_method_instance = method

        send_logs(self.extraction_identifier, performance_log)
        send_logs(self.extraction_identifier, f"Best method {best_method_instance.get_name()}")
        return best_method_instance

    def get_method_performance(
        self, method: PdfMultiOptionMethod, train_set: ExtractionData, test_set: ExtractionData
    ) -> float:
        method.set_parameters(train_set)

        if not method.can_be_used(train_set):
            send_logs(self.extraction_identifier, f"Not valid method {method.get_name()}")
            return 0

        send_logs(self.extraction_identifier, f"Checking {method.get_name()}")

        try:
            performance = method.get_performance(train_set, test_set)
        except Exception as e:
            severity = Severity.error if method.REPORT_ERRORS else Severity.info
            send_logs(self.extraction_identifier, f"Error checking {method.get_name()}", severity, e)
            performance = 0

        self.reset_extraction_data(train_set)

        send_logs(self.extraction_identifier, f"Performance {method.get_name()}: {round(performance, 2)}%")
        return performance

    def get_predictions_method(self):
        method_name = json.loads(self.method_name_path.read_text())
        for method in self.METHODS:
            if method.get_name() == method_name:
                return method

        return self.METHODS[0]

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        if not extraction_data.options and not extraction_data.extraction_identifier.get_options_path().exists():
            return False

        for sample in extraction_data.samples:
            if sample.pdf_data:
                return True

        return False

    @staticmethod
    def reset_extraction_data(multi_option_data: ExtractionData):
        for sample in multi_option_data.samples:
            for segment in sample.pdf_data.pdf_data_segments:
                segment.ml_label = 0

    @staticmethod
    def get_stats(extraction_data: ExtractionData):
        options = Counter()
        for sample in extraction_data.samples:
            options.update([option.label for option in sample.labeled_data.values])
        languages = Counter()
        for sample in extraction_data.samples:
            languages.update([sample.labeled_data.language_iso])

        options_count = len(extraction_data.options)
        stats = f"\nNumber of options: {options_count}\n"
        stats += f"Number of samples: {len(extraction_data.samples)}\n"
        stats += f"Languages\n"
        stats += "\n".join([f"{key} {value}" for key, value in languages.most_common()])
        stats += f"\nOptions\n"
        stats += "\n".join([f"{key} {value}" for key, value in options.most_common()])
        return stats
