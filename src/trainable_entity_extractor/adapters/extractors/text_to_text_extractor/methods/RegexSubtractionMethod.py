import re

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample

from tdda import *

from trainable_entity_extractor.adapters.extractors.ToTextExtractorMethod import ToTextExtractorMethod


class RegexSubtractionMethod(ToTextExtractorMethod):

    def train(self, extraction_data: ExtractionData):
        if not self.is_method_valid(extraction_data):
            return

        front_subtraction = [
            self.get_first_subtraction_characters(" ".join(x.get_input_text_by_lines()), x.labeled_data.label_text)
            for x in extraction_data.samples
        ][:500]
        front_regex_list = rexpy.extract([x for x in front_subtraction if x])
        front_regex_list = [regex[:-1] for regex in front_regex_list]

        back_subtraction = [
            self.get_last_subtraction_characters(" ".join(x.get_input_text_by_lines()), x.labeled_data.label_text)
            for x in extraction_data.samples
        ][:500]
        back_regex_list = rexpy.extract([x for x in back_subtraction if x])
        back_regex_list = [regex[1:] for regex in back_regex_list]

        self.save_json("regex_subtraction_list.json", front_regex_list + back_regex_list)

    def predict(self, predictions_samples: list[PredictionSample]) -> list[str]:
        regex_list = self.load_json("regex_subtraction_list.json")

        predictions = [" ".join(x.get_input_text_by_lines()) for x in predictions_samples]
        for i, prediction in enumerate(predictions):
            for regex in regex_list:
                matches = re.search(regex, prediction)
                if matches and not matches.start():
                    prediction = prediction[matches.end() :]
                    continue
                if matches and matches.end() == len(prediction):
                    prediction = prediction[: matches.start()]

            predictions[i] = prediction

        return predictions

    @staticmethod
    def get_first_subtraction_characters(segment_text: str, text: str):
        if text not in segment_text:
            return ""

        if text == segment_text:
            return ""

        first_index = segment_text.find(text)

        if not first_index:
            return ""

        return segment_text[:first_index]

    @staticmethod
    def get_last_subtraction_characters(segment_text: str, text: str):
        if text not in segment_text:
            return ""

        if text == segment_text:
            return ""

        first_index = segment_text.find(text) + len(text)

        if not first_index:
            return ""

        return segment_text[first_index:]

    def is_method_valid(self, extraction_data: ExtractionData) -> bool:
        front_subtraction = [
            self.get_first_subtraction_characters(" ".join(x.get_input_text_by_lines()), x.labeled_data.label_text)
            for x in extraction_data.samples[:20]
        ]
        front_regex_list = rexpy.extract([x for x in front_subtraction if x])

        back_subtraction = [
            self.get_last_subtraction_characters(" ".join(x.get_input_text_by_lines()), x.labeled_data.label_text)
            for x in extraction_data.samples[:20]
        ]
        back_regex_list = rexpy.extract([x for x in back_subtraction if x])

        if len(front_regex_list) > 4 or len(back_regex_list) > 4:
            return False

        return True
