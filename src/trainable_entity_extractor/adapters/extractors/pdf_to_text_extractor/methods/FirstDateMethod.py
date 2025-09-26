from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.adapters.extractors.ToTextExtractorMethod import ToTextExtractorMethod

from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.DateParserMethod import DateParserMethod


class FirstDateMethod(ToTextExtractorMethod):

    def train(self, extraction_data: ExtractionData):
        languages = [x.labeled_data.language_iso for x in extraction_data.samples]
        self.save_json("languages.json", list(set(languages)))

    def predict(self, prediction_samples_data: PredictionSamplesData) -> list[str]:
        predictions_samples = prediction_samples_data.prediction_samples
        predictions = [""] * len(predictions_samples)
        languages = self.load_json("languages.json")
        for index, prediction_sample in enumerate(predictions_samples):
            segments = prediction_sample.pdf_data.pdf_data_segments

            if predictions[index] or not prediction_sample.pdf_data or not segments:
                continue

            predictions[index] = self.get_date_from_segments(segments, languages)

        return predictions

    def get_date_from_segments(self, segments: list[PdfDataSegment], languages: list[str]) -> str:
        for segment in self.loop_segments(segments):
            date = DateParserMethod.get_date([segment.text_content], languages)
            if date:
                segment.ml_label = 1
                return date.strftime("%Y-%m-%d")

        return ""

    @staticmethod
    def loop_segments(segments: list[PdfDataSegment]):
        for segment in segments:
            yield segment
