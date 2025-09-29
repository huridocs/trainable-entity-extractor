import re

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData

from tdda import *

from trainable_entity_extractor.adapters.extractors.ToTextExtractorMethod import ToTextExtractorMethod


class RegexMethod(ToTextExtractorMethod):

    def train(self, extraction_data: ExtractionData):
        samples = [x.labeled_data.label_text for x in extraction_data.samples]
        samples = [sample for sample in samples if sample]
        regex_list = rexpy.extract(samples)
        regex_list = [regex[1:-1] for regex in regex_list]
        self.save_json("regex_list.json", regex_list)

    def predict(self, prediction_samples_data: PredictionSamplesData) -> list[str]:
        predictions = [""] * len(prediction_samples_data.prediction_samples)
        regex_list = self.load_json("regex_list.json")
        for regex in regex_list:
            for index, prediction_sample in enumerate(prediction_samples_data.prediction_samples):
                if predictions[index]:
                    break

                text = prediction_sample.get_input_text()

                match = re.search(regex, text)
                if match:
                    predictions[index] = str(match.group())

        return predictions
