import re

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.use_cases.extractors.ToTextExtractorMethod import ToTextExtractorMethod
from dateparser.search import search_dates


class DateParserMethod(ToTextExtractorMethod):

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
        self.save_json("languages.json", list(set(languages)))

    def predict(self, predictions_samples: list[PredictionSample]) -> list[str]:
        languages = self.load_json("languages.json")
        predictions_dates = [
            self.get_date(prediction_sample.get_input_text_by_lines(), languages)
            for prediction_sample in predictions_samples
        ]
        predictions = [date.strftime("%Y-%m-%d") if date else "" for date in predictions_dates]
        return predictions
