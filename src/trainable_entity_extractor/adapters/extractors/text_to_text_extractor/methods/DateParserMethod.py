import re

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.adapters.extractors.ToTextExtractorMethod import ToTextExtractorMethod
from dateparser.search import search_dates


class DateParserMethod(ToTextExtractorMethod):
    IS_VALID_EXECUTION_FILE_NAME = "date_parser_is_valid.json"

    DOTTED_DATE_PATTERN = re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b")
    DOTTED_DMY_LANGUAGES = {"ru", "uk", "pl", "sk", "bg", "be", "kk", "sr", "hr", "sl", "ro", "lt", "lv", "et", "cs", "fi"}

    @staticmethod
    def get_best_date(dates):
        if not dates:
            return None

        not_numbers_dates = [date for date in dates if re.search("[a-zA-Z]", date[0])]
        if not_numbers_dates:
            return not_numbers_dates[0][1]

        return dates[0][1]

    @staticmethod
    def has_dotted_date(text: str, languages) -> bool:
        if not languages or not any(lang in DateParserMethod.DOTTED_DMY_LANGUAGES for lang in languages):
            return False
        return bool(DateParserMethod.DOTTED_DATE_PATTERN.search(text))

    @staticmethod
    def get_date(tags_texts: list[str], languages):
        if not tags_texts:
            return ""
        text = " ".join(tags_texts)
        try:
            dates = search_dates(text, languages=languages)

            if DateParserMethod.has_dotted_date(text, languages):
                de_dates = list()
                for match in DateParserMethod.DOTTED_DATE_PATTERN.findall(text):
                    match_dates = search_dates(match, languages=["de"], settings={"DATE_ORDER": "DMY"})
                    if match_dates:
                        de_dates.extend(match_dates)
                if de_dates:
                    dates = de_dates + (dates or [])

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

    def predict(self, prediction_samples_data: PredictionSamplesData) -> list[str]:
        if self.load_json(self.IS_VALID_EXECUTION_FILE_NAME) == "false":
            return [""] * len(prediction_samples_data.prediction_samples)

        languages = self.load_json("languages.json")
        predictions_dates = [
            self.get_date(prediction_sample.get_input_text_by_lines(), languages)
            for prediction_sample in prediction_samples_data.prediction_samples
        ]

        return [prediction_date.strftime("%Y-%m-%d") if prediction_date else "" for prediction_date in predictions_dates]
