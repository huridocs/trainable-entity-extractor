from statistics import mode

from flair.nn import Classifier

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.adapters.extractors.ToTextExtractorMethod import ToTextExtractorMethod
from flair.data import Sentence

TAG_TYPE_JSON = "types.json"

tagger = Classifier.load("ner-ontonotes-large")


class NerFirstAppearanceMethod(ToTextExtractorMethod):
    def train(self, extraction_data: ExtractionData):
        texts = [self.clean_text(sample.get_input_text()) for sample in extraction_data.samples]
        labels = [self.clean_text(sample.labeled_data.label_text).lower() for sample in extraction_data.samples]

        types = list()

        for text, label in zip(texts, labels):
            sentence = Sentence(text)
            tagger.predict(sentence)
            label_types = [span.tag for span in sentence.get_spans() if label in self.clean_text(span.text).lower()]
            if label_types:
                types.append(label_types[0])

        self.save_json(TAG_TYPE_JSON, mode(types) if types else "")

    def predict(self, prediction_samples_data: PredictionSamplesData) -> list[str]:
        tag_type = self.load_json(TAG_TYPE_JSON)
        if not tag_type:
            return [""] * len(prediction_samples_data.prediction_samples)

        predictions = list()
        for prediction_sample in prediction_samples_data.prediction_samples:
            text = self.clean_text(prediction_sample.get_input_text())
            sentence = Sentence(text)
            tagger.predict(sentence)

            for span in sentence.get_spans():
                if span.tag == tag_type:
                    predictions.append(span.text)
                    break
            else:
                predictions.append("")

        return predictions

    @staticmethod
    def clean_text(text: str) -> str:
        return text.replace("\n", " ").replace("\t", " ").strip()
