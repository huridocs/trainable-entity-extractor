import re

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.adapters.extractors.ToTextExtractorMethod import ToTextExtractorMethod
from tdda import *


class PdfToTextRegexMethod(ToTextExtractorMethod):

    def train(self, extraction_data: ExtractionData):
        samples = [x.labeled_data.label_text for x in extraction_data.samples]
        samples = [sample for sample in samples if sample]
        regex_list = rexpy.extract(samples)
        regex_list = [regex[1:-1] for regex in regex_list]
        self.save_json("regex_list.json", regex_list)

    def predict(self, predictions_samples: list[PredictionSample]) -> list[str]:
        predictions = [""] * len(predictions_samples)
        regex_list = self.load_json("regex_list.json")
        for regex in regex_list:
            for index, prediction_sample in enumerate(predictions_samples):
                segments = prediction_sample.pdf_data.pdf_data_segments

                if predictions[index] or not prediction_sample.pdf_data or not segments:
                    break

                predictions[index] = self.get_matches(regex, segments)

        return predictions

    def get_matches(self, regex, segments):
        match = re.search(regex, self.clean_text(segments[0].text_content))
        if match:
            segments[0].ml_label = 1
            return str(match.group())

        for segment, next_segment in zip(segments, segments[1:]):
            text = self.clean_text(segment.text_content + " " + next_segment.text_content)
            match = re.search(regex, text)

            if match:
                segment.ml_label = 1
                next_segment.ml_label = 1
                return str(match.group())

        return ""
