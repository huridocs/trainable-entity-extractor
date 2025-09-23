from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.adapters.extractors.ToTextExtractorMethod import ToTextExtractorMethod
from trainable_entity_extractor.adapters.extractors.GlinerDateExtractor import GlinerDateExtractor


class GlinerDateParserMethod(ToTextExtractorMethod):
    gpu_needed = True
    IS_VALID_EXECUTION_FILE_NAME = "gliner_date_is_valid.json"

    @staticmethod
    def get_alphanumeric_text_with_spaces(text):
        return "".join([letter for letter in text if letter.isalnum() or letter.isspace()])

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
        gliner_date_extractor = GlinerDateExtractor()

        for sample in extraction_data.samples[:15]:
            if not sample.labeled_data.label_text.strip():
                continue
            dates = gliner_date_extractor.extract_dates(sample.labeled_data.label_text)
            if not dates:
                self.save_json(self.IS_VALID_EXECUTION_FILE_NAME, "false")
                return

        self.save_json(self.IS_VALID_EXECUTION_FILE_NAME, "true")

    def predict(self, prediction_samples_data: PredictionSamplesData) -> list[str]:
        if self.load_json(self.IS_VALID_EXECUTION_FILE_NAME) == "false":
            return [""] * len(prediction_samples_data.prediction_samples)

        predictions_dates = [
            self.get_date(prediction_sample.get_input_text_by_lines())
            for prediction_sample in prediction_samples_data.prediction_samples
        ]
        predictions = [date.strftime("%Y-%m-%d") if date else "" for date in predictions_dates]
        return predictions


if __name__ == "__main__":
    print(GlinerDateParserMethod.get_alphanumeric_text_with_spaces("21 DE MARÇO DE 2023"))
