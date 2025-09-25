from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FastSegmentSelectorFuzzy95 import (
    FastSegmentSelectorFuzzy95,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyCommas import (
    FuzzyCommas,
)


class FastSegmentSelectorFuzzyCommas(FastSegmentSelectorFuzzy95):
    def train(self, multi_option_data: ExtractionData):
        self.set_parameters(multi_option_data)
        super().train(multi_option_data)
        FuzzyCommas(self.extraction_identifier).train(multi_option_data)

    def predict(self, prediction_samples_data: PredictionSamplesData) -> list[list[Value]]:
        self.options = prediction_samples_data.options
        self.multi_value = prediction_samples_data.multi_value
        self.prediction_samples_data = self.get_prediction_data(prediction_samples_data)
        return FuzzyCommas(self.extraction_identifier).predict(self.prediction_samples_data)
