from trainable_entity_extractor.data.Option import Option
from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import (
    TextToMultiOptionMethod,
)


class NaiveTextToMultiOptionMethod(TextToMultiOptionMethod):

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        return True

    def predict(self, predictions_samples: list[PredictionSample]) -> list[list[Option]]:
        return [[] for _ in predictions_samples]

    def train(self, multi_option_data: ExtractionData):
        pass
