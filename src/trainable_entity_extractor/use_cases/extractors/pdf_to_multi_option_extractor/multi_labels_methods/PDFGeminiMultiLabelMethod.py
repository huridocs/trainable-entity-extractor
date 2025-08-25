from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.MultiLabelMethod import MultiLabelMethod
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.gemini_multi_option.TextGeminiMultiOption import (
    TextGeminiMultiOption,
)


class PDFGeminiMultiLabelMethod(MultiLabelMethod):
    def __init__(self, extraction_identifier, options, multi_value, method_name=""):
        super().__init__(extraction_identifier, options, multi_value, method_name)
        self._text_gemini = TextGeminiMultiOption(extraction_identifier, options, multi_value, method_name)

    def can_be_used(self, extraction_data):
        return self._text_gemini.can_be_used(extraction_data)

    @staticmethod
    def should_be_retrained_with_more_data():
        return False

    def train(self, multi_option_data: ExtractionData):
        training_samples: list[TrainingSample] = list()
        for sample in multi_option_data.samples:
            if not sample.pdf_data or not sample.labeled_data or not sample.labeled_data.values:
                continue
            text = sample.pdf_data.get_text().replace("\n", " ")
            training_samples.append(TrainingSample.from_values(text, sample.labeled_data.values))

        training_multi_option_data = ExtractionData(
            samples=training_samples,
            extraction_identifier=self.extraction_identifier,
            options=self.options,
        )
        return self._text_gemini.train(training_multi_option_data)

    def predict(self, multi_option_data: ExtractionData) -> list[list[Value]]:
        texts = [sample.pdf_data.get_text() for sample in multi_option_data.samples]
        texts = [text.replace("\n", " ") for text in texts]

        prediction_samples = [PredictionSample.from_text(text) for text in texts]
        options_list = self._text_gemini.predict(prediction_samples)
        return [[Value.from_option(option) for option in options] if options else [] for options in options_list]
