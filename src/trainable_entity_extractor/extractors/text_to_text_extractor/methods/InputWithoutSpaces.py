from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.extractors.ToTextExtractorMethod import ToTextExtractorMethod


class InputWithoutSpaces(ToTextExtractorMethod):

    def train(self, extraction_data: ExtractionData):
        self.save_json("best_method.json", True)

    @staticmethod
    def trim_text(tag_texts: list[str]) -> str:
        text = "".join(tag_texts)
        return "".join(text.split())

    def predict(self, predictions_samples: list[PredictionSample]) -> list[str]:
        return [self.trim_text(x.tags_texts) for x in predictions_samples]
