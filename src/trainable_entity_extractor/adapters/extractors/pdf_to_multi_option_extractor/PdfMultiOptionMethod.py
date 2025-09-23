import shutil
from os.path import join
from typing import Type

from sklearn.metrics import f1_score

from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.MultiLabelMethod import MultiLabelMethod
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.FilterSegmentsMethod import (
    FilterSegmentsMethod,
)
from trainable_entity_extractor.ports.MethodBase import MethodBase


class PdfMultiOptionMethod(MethodBase):
    REPORT_ERRORS = True

    def __init__(
        self,
        filter_segments_method: Type[FilterSegmentsMethod] = None,
        multi_label_method: Type[MultiLabelMethod] = None,
    ):
        # Initialize with a default extraction_identifier - will be updated via set_parameters
        super().__init__(ExtractionIdentifier(run_name="not set", extraction_name="not set"))
        self.filter_segments_method = filter_segments_method
        self.multi_label_method = multi_label_method
        self.options: list[Option] = list()
        self.multi_value = False
        self.extraction_data = None
        self.prediction_samples_data = None

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
        multi_label = self.multi_label_method(self.extraction_identifier, self.get_name())
        multi_label.train(filtered_multi_option_data)

    def predict(self, prediction_samples_data: PredictionSamplesData) -> list[list[Value]]:
        self.options = prediction_samples_data.options
        self.multi_value = prediction_samples_data.multi_value
        self.prediction_samples_data = prediction_samples_data

        print("Filtering segments")
        prediction_samples_data = self.filter_segments_method().filter_prediction_samples(prediction_samples_data)

        print("Prediction")
        multi_label = self.multi_label_method(self.extraction_identifier, self.get_name())
        predictions = multi_label.predict(prediction_samples_data)

        return predictions

    def get_samples_for_context(self, prediction_samples_data: PredictionSamplesData) -> list[PredictionSample]:
        if self.prediction_samples_data:
            return self.prediction_samples_data.prediction_samples

        return prediction_samples_data.prediction_samples

    def can_be_used(self, multi_option_data: ExtractionData) -> bool:
        if self.multi_label_method:
            multi_label = self.multi_label_method(self.extraction_identifier)
            return multi_label.can_be_used(multi_option_data)

        return True

    def should_be_retrained_with_more_data(self):
        if self.multi_label_method:
            return self.multi_label_method.should_be_retrained_with_more_data()
        return True

    def remove_method_data(self) -> None:
        shutil.rmtree(join(self.extraction_identifier.get_path(), self.get_name()), ignore_errors=True)
