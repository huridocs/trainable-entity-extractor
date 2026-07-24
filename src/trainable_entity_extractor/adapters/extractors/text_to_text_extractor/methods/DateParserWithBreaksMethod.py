from dateparser.search import search_dates

from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.DateParserMethod import DateParserMethod


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

            if DateParserMethod.has_dotted_date(text, languages):
                de_dates = list()
                for match in DateParserMethod.DOTTED_DATE_PATTERN.findall(text):
                    match_dates = search_dates(match, languages=["de"], settings={"DATE_ORDER": "DMY"})
                    if match_dates:
                        de_dates.extend(match_dates)
                if de_dates:
                    dates = de_dates + (dates or [])

            if not dates:
                dates = search_dates(text_with_breaks)

            return DateParserMethod.get_best_date(dates)

        except TypeError:
            return None
        except IndexError:
            return None
