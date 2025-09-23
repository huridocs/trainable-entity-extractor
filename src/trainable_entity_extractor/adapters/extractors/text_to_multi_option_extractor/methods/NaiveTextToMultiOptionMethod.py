from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import (
    TextToMultiOptionMethod,
)


class NaiveTextToMultiOptionMethod(TextToMultiOptionMethod):

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        return True

    def predict(self, prediction_samples_data: PredictionSamplesData) -> list[list[Option]]:
        predictions: list[list[Option]] = list()
        for _ in prediction_samples_data.prediction_samples:
            predictions.append(prediction_samples_data.options[:1])

        return predictions

    def train(self, multi_option_data: ExtractionData):
        pass
