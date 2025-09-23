from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.adapters.extractors.ToTextExtractorMethod import ToTextExtractorMethod


class InputWithoutSpaces(ToTextExtractorMethod):

    def train(self, extraction_data: ExtractionData):
        self.save_json("best_method.json", True)

    @staticmethod
    def trim_text(tag_texts: list[str]) -> str:
        if not tag_texts:
            return ""
        text = "".join(tag_texts)
        return "".join(text.split())

    def predict(self, prediction_samples_data: PredictionSamplesData) -> list[str]:
        return [self.trim_text(x.get_input_text_by_lines()) for x in prediction_samples_data.prediction_samples]
