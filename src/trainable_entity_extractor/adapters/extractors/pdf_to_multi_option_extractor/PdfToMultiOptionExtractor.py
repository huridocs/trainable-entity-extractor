from collections import Counter
from os.path import join
from typing import Optional

from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.ports.ExtractorBase import ExtractorBase
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.PdfMultiOptionMethod import (
    PdfMultiOptionMethod,
)

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanBeginningDotDigits1000 import (
    CleanBeginningDotDigits1000,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanBeginningDotDigits500 import (
    CleanBeginningDotDigits500,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanEndDotDigits1000 import (
    CleanEndDotDigits1000,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_labels_methods.FastTextMethod import (
    FastTextMethod,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_labels_methods.PDFGeminiMultiLabelMethod import (
    PDFGeminiMultiLabelMethod,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_labels_methods.SetFitEnglishMethod import (
    SetFitEnglishMethod,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_labels_methods.SetFitMultilingualMethod import (
    SetFitMultilingualMethod,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_labels_methods.SingleLabelSetFitEnglishMethod import (
    SingleLabelSetFitEnglishMethod,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_labels_methods.SingleLabelSetFitMultilingualMethod import (
    SingleLabelSetFitMultilingualMethod,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FastSegmentSelectorFuzzy95 import (
    FastSegmentSelectorFuzzy95,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FastSegmentSelectorFuzzyCommas import (
    FastSegmentSelectorFuzzyCommas,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyAll100 import (
    FuzzyAll100,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyAll75 import (
    FuzzyAll75,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyAll88 import (
    FuzzyAll88,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyFirst import (
    FuzzyFirst,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyFirstCleanLabel import (
    FuzzyFirstCleanLabel,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyLast import (
    FuzzyLast,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyLastCleanLabel import (
    FuzzyLastCleanLabel,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzySegmentSelector import (
    FuzzySegmentSelector,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.NextWordsTokenSelectorFuzzy75 import (
    NextWordsTokenSelectorFuzzy75,
)

from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.PreviousWordsSentenceSelectorFuzzyCommas import (
    PreviousWordsSentenceSelectorFuzzyCommas,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.PreviousWordsTokenSelectorFuzzy75 import (
    PreviousWordsTokenSelectorFuzzy75,
)
from trainable_entity_extractor.adapters.extractors.segment_selector.FastAndPositionsSegmentSelector import (
    FastAndPositionsSegmentSelector,
)
from trainable_entity_extractor.adapters.extractors.segment_selector.FastSegmentSelector import FastSegmentSelector
from trainable_entity_extractor.adapters.extractors.segment_selector.SegmentSelector import SegmentSelector
from trainable_entity_extractor.ports.Logger import Logger

RETRAIN_SAMPLES_THRESHOLD = 250


class PdfToMultiOptionExtractor(ExtractorBase):
    METHODS: list[PdfMultiOptionMethod] = [
        FuzzyFirst(),
        FuzzyLast(),
        FuzzyFirstCleanLabel(),
        FuzzyLastCleanLabel(),
        FuzzyAll100(),
        FuzzyAll88(),
        FuzzyAll75(),
        PreviousWordsTokenSelectorFuzzy75(),
        NextWordsTokenSelectorFuzzy75(),
        PreviousWordsSentenceSelectorFuzzyCommas(),
        FastSegmentSelectorFuzzy95(),
        FastSegmentSelectorFuzzyCommas(),
        FuzzySegmentSelector(),
        PdfMultiOptionMethod().set_methods(CleanBeginningDotDigits500, FastTextMethod),
        PdfMultiOptionMethod().set_methods(CleanEndDotDigits1000, FastTextMethod),
        PdfMultiOptionMethod().set_methods(CleanEndDotDigits1000, PDFGeminiMultiLabelMethod),
        PdfMultiOptionMethod().set_methods(CleanBeginningDotDigits1000, PDFGeminiMultiLabelMethod),
        PdfMultiOptionMethod().set_methods(CleanBeginningDotDigits1000, SetFitEnglishMethod),
        PdfMultiOptionMethod().set_methods(CleanBeginningDotDigits1000, SetFitMultilingualMethod),
        PdfMultiOptionMethod().set_methods(CleanBeginningDotDigits1000, SingleLabelSetFitEnglishMethod),
        PdfMultiOptionMethod().set_methods(CleanBeginningDotDigits1000, SingleLabelSetFitMultilingualMethod),
    ]

    def __init__(self, extraction_identifier: ExtractionIdentifier, logger: Logger):
        super().__init__(extraction_identifier, logger)
        self.base_path = join(self.extraction_identifier.get_path(), "multi_option_extractor")
        self.options: list[Option] = list()
        self.multi_value = False

    def prepare_for_training(self, extraction_data: ExtractionData) -> tuple[ExtractionData, ExtractionData]:
        self.options = extraction_data.options
        self.multi_value = extraction_data.multi_value
        SegmentSelector(self.extraction_identifier).prepare_model_folder()
        FastSegmentSelector(self.extraction_identifier).prepare_model_folder()
        FastAndPositionsSegmentSelector(self.extraction_identifier).prepare_model_folder()

        return ExtractorBase.get_train_test_sets(extraction_data)

    def get_suggestions(self, method_name: str, prediction_samples_data: PredictionSamplesData) -> list[Suggestion]:
        if not prediction_samples_data:
            return []

        prediction_samples, predictions = self.get_predictions(method_name, prediction_samples_data)
        prediction_method = self.get_method_instance_by_name(method_name)

        use_context_from_the_end = "End" in prediction_method.get_name()
        suggestions = list()
        for prediction_sample, prediction in zip(prediction_samples, predictions):
            suggestion = Suggestion.get_empty(self.extraction_identifier, prediction_sample.entity_name)
            suggestion.add_prediction_multi_option(prediction_sample, prediction, use_context_from_the_end)
            suggestion.xml_file_name = prediction_sample.entity_name
            suggestions.append(suggestion)

        return suggestions

    def get_predictions(
        self, method_name: str, prediction_samples_data: PredictionSamplesData
    ) -> tuple[list[PredictionSample], list[list[Value]]]:
        self.options = prediction_samples_data.options
        self.multi_value = prediction_samples_data.multi_value

        method = self.get_method_instance_by_name(method_name)
        self.logger.log(self.extraction_identifier, f"Using method {method.get_name()} for suggestions")

        prediction = method.predict(prediction_samples_data=prediction_samples_data)

        if not self.multi_value:
            prediction = [x[:1] for x in prediction]

        return method.get_samples_for_context(prediction_samples_data), prediction

    def get_method_performance(
        self, method: PdfMultiOptionMethod, train_set: ExtractionData, test_set: ExtractionData
    ) -> float:
        method.set_parameters(train_set)

        if not method.can_be_used(train_set):
            self.logger.log(self.extraction_identifier, f"Not valid method {method.get_name()}")
            return 0

        self.logger.log(self.extraction_identifier, f"Checking {method.get_name()}")

        try:
            performance = method.get_performance(train_set, test_set)
        except Exception as e:
            severity = LogSeverity.error if method.REPORT_ERRORS else LogSeverity.info
            self.logger.log(self.extraction_identifier, f"Error checking {method.get_name()}", severity, e)
            performance = 0

        self.reset_extraction_data(train_set)
        return performance

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        if not extraction_data.options:
            return False

        for sample in extraction_data.samples:
            if sample.pdf_data and sample.pdf_data.contains_text():
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

        empty_pdfs = [
            x for x in extraction_data.samples if x.pdf_data and x.pdf_data.pdf_features and not x.pdf_data.contains_text()
        ]
        options_count = len(extraction_data.options)
        stats = f"\nNumber of options: {options_count}\n"
        stats += f"Number of samples: {len(extraction_data.samples)}\n"
        stats += f"Empty PDFs: {len(empty_pdfs)}\n" if empty_pdfs else ""
        stats += f"Languages\n"
        stats += "\n".join([f"{key} {value}" for key, value in languages.most_common()])
        stats += f"\nOptions\n"
        stats += "\n".join([f"{key} {value}" for key, value in options.most_common()])
        return stats
