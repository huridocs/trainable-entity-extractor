from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.MultiLabelMethod import MultiLabelMethod
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.gemini_multi_option.TextGeminiMultiOption import (
    TextGeminiMultiOption,
)


class GeminiMultiLabelMethod(MultiLabelMethod):
    def __init__(self, extraction_identifier, options, multi_value, method_name=""):
        super().__init__(extraction_identifier, options, multi_value, method_name)
        self._text_gemini = TextGeminiMultiOption(extraction_identifier, options, multi_value)

    def can_be_used(self, extraction_data):
        return self._text_gemini.can_be_used(extraction_data)

    def train(self, multi_option_data: ExtractionData):
        return self._text_gemini.train(multi_option_data)

    def predict(self, multi_option_data: ExtractionData) -> list[list[Value]]:
        prediction_samples = [PredictionSample.from_pdf_data(sample.pdf_data) for sample in multi_option_data.samples]
        options_list = self._text_gemini.predict(prediction_samples)
        return [[Value.from_option(option) for option in options] if options else [] for options in options_list]
