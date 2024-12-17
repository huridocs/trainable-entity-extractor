from statistics import mode

from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.extractors.ToTextExtractorMethod import ToTextExtractorMethod
from flair.data import Sentence
from flair.nn import Classifier

TAG_TYPE_JSON = "types.json"

# tagger = Classifier.load("ner-ontonotes-large")


class NerFirstAppearanceMethod(ToTextExtractorMethod):
    def train(self, extraction_data: ExtractionData):
        texts = [self.clean_text(" ".join(sample.segment_selector_texts)) for sample in extraction_data.samples]
        labels = [self.clean_text(sample.labeled_data.label_text).lower() for sample in extraction_data.samples]

        types = list()

        for text, label in zip(texts, labels):
            sentence = Sentence(text)
            tagger.predict(sentence)
            label_types = [span.tag for span in sentence.get_spans() if label in self.clean_text(span.text).lower()]
            if label_types:
                types.append(label_types[0])

        self.save_json(TAG_TYPE_JSON, mode(types) if types else "")

    def predict(self, predictions_samples: list[PredictionSample]) -> list[str]:
        tag_type: str = self.load_json(TAG_TYPE_JSON)
        if not tag_type:
            return [""] * len(predictions_samples)

        texts = [self.clean_text(" ".join(sample.segment_selector_texts)) for sample in predictions_samples]
        predictions = [""] * len(texts)
        for i, text in enumerate(texts):
            sentence = Sentence(text)
            tagger.predict(sentence)
            prediction_texts = [span.text for span in sentence.get_spans() if span.tag == tag_type]
            predictions[i] = self.get_appearance(prediction_texts)

        return predictions

    def get_appearance(self, prediction_texts):
        return prediction_texts[0] if prediction_texts else ""
