from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.extractors.ToTextExtractorMethod import ToTextExtractorMethod


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
        return [self.trim_text(x.segment_selector_texts) for x in predictions_samples]
