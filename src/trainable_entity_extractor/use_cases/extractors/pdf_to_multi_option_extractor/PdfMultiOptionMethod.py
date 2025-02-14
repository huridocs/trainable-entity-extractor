import shutil
from os.path import join
from typing import Type

from sklearn.metrics import f1_score

from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.MultiLabelMethod import MultiLabelMethod
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.FilterSegmentsMethod import (
    FilterSegmentsMethod,
)


class PdfMultiOptionMethod:
    REPORT_ERRORS = True

    def __init__(
        self,
        filter_segments_method: Type[FilterSegmentsMethod] = None,
        multi_label_method: Type[MultiLabelMethod] = None,
    ):
        self.multi_label_method = multi_label_method
        self.filter_segments_method = filter_segments_method
        self.extraction_identifier = ExtractionIdentifier(run_name="not set", extraction_name="not set")
        self.options: list[Option] = list()
        self.multi_value = False
        self.extraction_data = None

    def set_parameters(self, multi_option_data: ExtractionData):
        self.extraction_identifier = multi_option_data.extraction_identifier
        self.options = multi_option_data.options
        self.multi_value = multi_option_data.multi_value
        self.extraction_data = multi_option_data

    def get_name(self):
        if self.filter_segments_method and self.multi_label_method:
            text_extractor_name = self.filter_segments_method.__name__.replace("Method", "")
            multi_option_name = self.multi_label_method.__name__.replace("Method", "")
            text_extractor_name = text_extractor_name.replace("TextAtThe", "")
            multi_option_name = multi_option_name.replace("TextAtThe", "")
            return f"{text_extractor_name}_{multi_option_name}"

        return self.__class__.__name__

    def get_performance(self, train_set: ExtractionData, test_set: ExtractionData) -> float:
        self.set_parameters(train_set)
        truth_one_hot = self.one_hot_to_options_list([x.labeled_data.values for x in test_set.samples], self.options)

        self.train(train_set)
        predictions = self.predict(test_set)

        if not self.multi_value:
            predictions = [x[:1] for x in predictions]

        predictions_one_hot = self.one_hot_to_options_list(predictions, self.options)

        try:
            score = f1_score(truth_one_hot, predictions_one_hot, average="micro")
        except ValueError:
            score = 0

        return 100 * score

    @staticmethod
    def one_hot_to_options_list(pdfs_options: list[list[Option]], options: list[Option]) -> list[list[int]]:
        options_one_hot: list[list[int]] = list()
        option_labels = [x.label for x in options]
        for pdf_options in pdfs_options:
            pdf_options_one_hot = [0] * len(options)

            for pdf_option in pdf_options:
                if pdf_option.label in option_labels:
                    pdf_options_one_hot[option_labels.index(pdf_option.label)] = 1

            options_one_hot.append(pdf_options_one_hot)

        return options_one_hot

    def train(self, multi_option_data: ExtractionData):
        self.set_parameters(multi_option_data)

        print("Filtering segments")
        filtered_multi_option_data = self.filter_segments_method().filter(multi_option_data)

        print("Creating model")
        multi_label = self.multi_label_method(self.extraction_identifier, self.options, self.multi_value, self.get_name())
        multi_label.train(filtered_multi_option_data)

    def predict(self, multi_option_data: ExtractionData) -> list[list[Option]]:
        self.set_parameters(multi_option_data)

        print("Filtering segments")
        filtered_multi_option_data = self.filter_segments_method().filter(multi_option_data)

        print("Prediction")
        multi_label = self.multi_label_method(self.extraction_identifier, self.options, self.multi_value, self.get_name())
        predictions = multi_label.predict(filtered_multi_option_data)

        return predictions

    def get_samples_for_context(self, extraction_data: ExtractionData) -> list[TrainingSample]:
        if self.extraction_data:
            return self.extraction_data.samples

        return extraction_data.samples

    def can_be_used(self, multi_option_data: ExtractionData) -> bool:
        if self.multi_label_method:
            multi_label = self.multi_label_method(self.extraction_identifier, self.options, self.multi_value)
            return multi_label.can_be_used(multi_option_data)

        return True

    def remove_method_data(self, extraction_identifier: ExtractionIdentifier):
        shutil.rmtree(join(extraction_identifier.get_path(), self.get_name()), ignore_errors=True)
