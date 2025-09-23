from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyAll75 import (
    FuzzyAll75,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.NextWordsTokenSelectorFuzzy75 import (
    NextWordsTokenSelectorFuzzy75,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.PreviousWordsSegmentSelector import (
    PreviousWordsSegmentSelector,
)


class PreviousWordsTokenSelectorFuzzy75(NextWordsTokenSelectorFuzzy75):
    threshold = 75

    def predict(self, prediction_samples_data: PredictionSamplesData) -> list[list[Value]]:
        self.options = prediction_samples_data.options
        self.multi_value = prediction_samples_data.multi_value
        self.get_token_prediction_data(prediction_samples_data)
        segment_selector = PreviousWordsSegmentSelector(self.extraction_identifier)

        for sample in self.prediction_samples_data.prediction_samples:
            sample.pdf_data.pdf_data_segments = segment_selector.predict(sample.pdf_data.pdf_data_segments)
            self.mark_segments_for_context(sample.pdf_data.pdf_data_segments)

        return FuzzyAll75().predict(self.prediction_samples_data)

    def train(self, multi_option_data: ExtractionData):
        self.set_parameters(multi_option_data)
        self.get_token_extraction_data(multi_option_data)
        marked_segments = list()
        for sample in self.extraction_data.samples:
            marked_segments.extend(self.get_marked_segments(sample))

        PreviousWordsSegmentSelector(self.extraction_identifier).create_model(marked_segments)
