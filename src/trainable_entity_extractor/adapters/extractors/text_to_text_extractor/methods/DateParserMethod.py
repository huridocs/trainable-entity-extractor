import re

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSamples import PredictionSamples
from trainable_entity_extractor.adapters.extractors.ToTextExtractorMethod import ToTextExtractorMethod
from dateparser.search import search_dates


class DateParserMethod(ToTextExtractorMethod):
    IS_VALID_EXECUTION_FILE_NAME = "date_parser_is_valid.json"

    @staticmethod
    def get_best_date(dates):
        if not dates:
            return None

        not_numbers_dates = [date for date in dates if re.search("[a-zA-Z]", date[0])]
        if not_numbers_dates:
            return not_numbers_dates[0][1]

        return dates[0][1]

    @staticmethod
    def get_date(tags_texts: list[str], languages):
        if not tags_texts:
            return ""
        text = " ".join(tags_texts)
        try:
            dates = search_dates(text, languages=languages)

            if not dates:
                dates = search_dates(text)

            return DateParserMethod.get_best_date(dates)

        except TypeError:
            return None
        except IndexError:
            return None

    def train(self, extraction_data: ExtractionData):
        languages = [x.labeled_data.language_iso for x in extraction_data.samples]

        for sample in extraction_data.samples[:15]:
            if not sample.labeled_data.label_text.strip():
                continue
            date = self.get_date([sample.labeled_data.label_text], languages)
            if not date:
                self.save_json(self.IS_VALID_EXECUTION_FILE_NAME, "false")
                return

        self.save_json(self.IS_VALID_EXECUTION_FILE_NAME, "true")
        self.save_json("languages.json", list(set(languages)))

    def predict(self, prediction_samples: PredictionSamples) -> list[str]:
        if self.load_json(self.IS_VALID_EXECUTION_FILE_NAME) == "false":
            return [""] * len(prediction_samples.prediction_samples)

        languages = self.load_json("languages.json")
        predictions_dates = [
            self.get_date(prediction_sample.get_input_text_by_lines(), languages)
            for prediction_sample in prediction_samples.prediction_samples
        ]

        return [str(prediction_date) if prediction_date else "" for prediction_date in predictions_dates]
