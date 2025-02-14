from dateparser.search import search_dates

from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.DateParserMethod import DateParserMethod


class DateParserWithBreaksMethod(DateParserMethod):
    @staticmethod
    def get_date(tags_texts: list[str], languages):
        if not tags_texts:
            return ""
        text = " ".join(tags_texts)
        text_with_breaks = "\n".join([text for text in tags_texts])

        try:
            dates = search_dates(text_with_breaks, languages=languages)
            dates_without_breaks = search_dates(text, languages=languages)

            if not dates:
                dates = list()

            if dates_without_breaks:
                dates.extend(dates_without_breaks)

            if not dates:
                dates = search_dates(text_with_breaks)

            return DateParserMethod.get_best_date(dates)

        except TypeError:
            return None
        except IndexError:
            return None
