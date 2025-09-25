from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.Suggestion import Suggestion
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

    def get_suggestions(self, method_name: str, prediction_samples: PredictionSamplesData) -> list[Suggestion]:
        method_instance = self.get_method_instance_by_name(method_name)
        self.logger.log(
            self.extraction_identifier,
            f"And also using {method_instance.get_name()} to calculate {len(prediction_samples.prediction_samples)} suggestions",
        )
        prediction = method_instance.predict(prediction_samples)
        suggestions = list()
        for prediction_text, prediction_sample in zip(prediction, prediction_samples.prediction_samples):
            entity_name = prediction_sample.entity_name
            suggestion = Suggestion.from_prediction_text(self.extraction_identifier, entity_name, prediction_text)
            suggestion.set_segment_text_from_sample(prediction_sample)
            suggestions.append(suggestion)

        return suggestions
