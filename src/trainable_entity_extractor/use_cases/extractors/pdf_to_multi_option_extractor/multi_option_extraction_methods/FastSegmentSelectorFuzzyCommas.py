from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FastSegmentSelectorFuzzy95 import (
    FastSegmentSelectorFuzzy95,
)
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyCommas import (
    FuzzyCommas,
)


class FastSegmentSelectorFuzzyCommas(FastSegmentSelectorFuzzy95):
    def train(self, multi_option_data: ExtractionData):
        self.set_parameters(multi_option_data)
        super().train(multi_option_data)
        FuzzyCommas().train(multi_option_data)

    def predict(self, multi_option_data: ExtractionData) -> list[list[Option]]:
        self.set_parameters(multi_option_data)
        self.extraction_data = self.get_prediction_data(multi_option_data)
        return FuzzyCommas().predict(self.extraction_data)
