from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.use_cases.extractors.ToTextExtractorMethod import ToTextExtractorMethod


class SameInputOutputMethod(ToTextExtractorMethod):

    def train(self, extraction_data: ExtractionData):
        pass

    @staticmethod
    def trim_text(tag_texts: list[str]) -> str:
        if not tag_texts:
            return ""
        text = " ".join(tag_texts)
        return " ".join(text.split())

    def predict(self, predictions_samples: list[PredictionSample]) -> list[str]:
        return [self.trim_text(x.get_input_text_by_lines()) for x in predictions_samples]
