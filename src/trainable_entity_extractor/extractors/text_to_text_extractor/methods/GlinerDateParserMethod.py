import re

from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.extractors.ToTextExtractorMethod import ToTextExtractorMethod
from trainable_entity_extractor.extractors.GlinerDateExtractor import GlinerDateExtractor


class GlinerDateParserMethod(ToTextExtractorMethod):

    @staticmethod
    def get_alphanumeric_text_with_spaces(text):
        alphanumeric_pattern = re.compile(r"[A-Za-z0-9 ]+")
        return "".join(alphanumeric_pattern.findall(text))

    @staticmethod
    def get_date(tags_texts: list[str]):
        if not tags_texts:
            return ""
        text = GlinerDateParserMethod.get_alphanumeric_text_with_spaces(" ".join(tags_texts))
        try:
            gliner_date_extractor = GlinerDateExtractor()
            dates = gliner_date_extractor.extract_dates(text)
            return dates[0]
        except:
            pass

        return None

    def train(self, extraction_data: ExtractionData):
        pass

    def predict(self, predictions_samples: list[PredictionSample]) -> list[str]:
        predictions_dates = [
            self.get_date(prediction_sample.segment_selector_texts) for prediction_sample in predictions_samples
        ]
        predictions = [date.strftime("%Y-%m-%d") if date else "" for date in predictions_dates]
        return predictions
