import json
import os
import re
from os.path import join
from pathlib import Path

import rapidfuzz
from rapidfuzz import fuzz

from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.PdfMultiOptionMethod import (
    PdfMultiOptionMethod,
)

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.Appearance import (
    Appearance,
)


class FuzzyCommas(PdfMultiOptionMethod):
    threshold = 92

    def __init__(self):
        super().__init__()
        self.options_cleaned: list[str] = list()
        self.options_cleaned_by_length: list[str] = list()
        self.options_cleaned_words_sorted: list[str] = list()
        self.options_cleaned_words_sorted_by_length: list[str] = list()

    def get_appearances_for_segments(
        self, pdf_segments: list[PdfDataSegment], aliases: dict[str, list[str]]
    ) -> tuple[list[Appearance], list[str]]:
        appearances: list[Appearance] = []
        not_found_texts = []

        for pdf_segment in pdf_segments:
            text = pdf_segment.text_content
            texts_separated_by_comma = [
                cleaned_text for piece in re.split(",|:| and ", text) if (cleaned_text := self.clean_text(piece, False))
            ]

            for one_piece_text in texts_separated_by_comma:
                appearance = self.get_appearances_one_segment(one_piece_text, aliases)

                if appearance:
                    if appearance not in appearances:
                        appearances.append(Appearance(option_label=appearance, context=pdf_segment.text_content))
                else:
                    not_found_texts.append(one_piece_text)

        return appearances, not_found_texts

    def get_appearances_one_segment(self, text: str, aliases: dict[str, list[str]]) -> str:
        cleaned_text = self.clean_text(text, True)
        for option_cleaned in self.options_cleaned_words_sorted_by_length:
            if len(text) < len(option_cleaned) * 0.92 or len(text) > len(option_cleaned) * 1.2:
                continue

            if fuzz.partial_ratio(option_cleaned, cleaned_text) >= self.threshold:
                return self.options_cleaned[self.options_cleaned_words_sorted.index(option_cleaned)]

        for option_cleaned in self.options_cleaned_by_length:
            if option_cleaned not in aliases:
                continue

            for alias in aliases[option_cleaned]:
                if rapidfuzz.fuzz.ratio(alias, text) > self.threshold:
                    return option_cleaned

        return ""

    @staticmethod
    def clean_text(text: str, sort_words: bool) -> str:
        text = text.lower()
        text = "".join([letter for letter in text if letter.isalnum() or letter == " "])

        if sort_words:
            text = " ".join(sorted(text.split()))
        else:
            text = " ".join(text.split())

        return text

    def clean_texts(self, texts: list[str], sort_words: bool) -> list[str]:
        return list([self.clean_text(option, sort_words) for option in texts])

    def predict(self, multi_option_data: ExtractionData) -> list[list[Value]]:
        self.set_parameters(multi_option_data)
        self.set_options_variants()

        try:
            aliases = json.loads(self.get_aliases_path().read_text())
            if not aliases or not isinstance(aliases, dict):
                raise FileNotFoundError

        except FileNotFoundError:
            aliases = dict()

        predictions = list()

        for multi_option_sample in multi_option_data.samples:
            pdf_segments: list[PdfDataSegment] = [x for x in multi_option_sample.pdf_data.pdf_data_segments]
            appearances, _ = self.get_appearances_for_segments(pdf_segments, aliases)
            prediction_options = [x.to_value(self.options_cleaned, self.options) for x in appearances]
            predictions.append(prediction_options)

        return predictions

    def train(self, multi_option_data: ExtractionData):
        self.set_parameters(multi_option_data)
        self.set_options_variants()

        aliases: dict[str, list[str]] = {option: list() for option in self.options_cleaned}
        for sample in multi_option_data.samples:
            sample_aliases = self.get_aliases(sample)
            for option, sample_alias in sample_aliases.items():
                aliases[option] = list(dict.fromkeys(aliases[option] + [sample_alias]))

        self.get_aliases_path().write_text(json.dumps(aliases))

    def get_aliases_path(self) -> Path:
        path = Path(join(self.extraction_identifier.get_path(), "fuzzy_commas"))

        if not path.exists():
            os.makedirs(path, exist_ok=True)

        return Path(join(path, "aliases.json"))

    def get_aliases(self, sample: TrainingSample) -> dict[str, str]:
        segments = [segment for segment in sample.pdf_data.pdf_data_segments if segment.ml_label]
        appearances, not_found_texts = self.get_appearances_for_segments(segments, {})
        values_ids = {option.id for option in sample.labeled_data.values}
        truth_options = self.clean_texts([option.label for option in self.options if option.id in values_ids], False)
        not_found_options = [option for option in truth_options if option not in appearances]
        return self.find_aliases(not_found_options, not_found_texts)

    @staticmethod
    def find_aliases(not_found_options: list[str], not_found_texts: list[str]) -> dict[str, str]:
        aliases = dict()
        cleaned_texts = [" ".join(text.lower().strip().split()) for text in not_found_texts]

        for option in not_found_options:
            for text in cleaned_texts:
                if rapidfuzz.fuzz.partial_ratio(option, text) > 80:
                    aliases[option] = text

        return aliases

    def set_options_variants(self):
        self.options_cleaned = self.clean_texts(texts=[x.label for x in self.options], sort_words=False)
        self.options_cleaned_by_length = sorted(self.options_cleaned, key=lambda x: -len(x))
        self.options_cleaned_words_sorted = self.clean_texts(texts=[x.label for x in self.options], sort_words=True)
        self.options_cleaned_words_sorted_by_length = sorted(self.options_cleaned_words_sorted, key=lambda x: -len(x))
