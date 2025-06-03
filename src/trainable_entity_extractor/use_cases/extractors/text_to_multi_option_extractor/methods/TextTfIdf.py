import os
from functools import lru_cache
from os.path import join, exists

import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier

from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import (
    TextToMultiOptionMethod,
)
from joblib import dump, load

nltk.download("wordnet")
nltk.download("stopwords")

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words("english")
stop_words_set = set(stop_words)

lemmatize = lru_cache(maxsize=50000)(lemmatizer.lemmatize)


class TextTfIdf(TextToMultiOptionMethod):

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        return True

    def get_data_path(self):
        model_folder_path = join(self.extraction_identifier.get_path(), self.get_name())

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        return join(model_folder_path, "data.txt")

    def get_model_path(self):
        model_folder_path = join(self.extraction_identifier.get_path(), self.get_name())

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        return join(model_folder_path, "fast.model")

    def train(self, multi_option_data: ExtractionData):
        texts = [sample.get_input_text() for sample in multi_option_data.samples]
        dump(texts, self.get_data_path())

        vectorized = TfidfVectorizer()
        tfidf_train_vectors = vectorized.fit_transform(texts)

        labels = self.get_one_hot_encoding(multi_option_data)
        one_vs_rest_classifier = OneVsRestClassifier(RandomForestClassifier())
        one_vs_rest_classifier = one_vs_rest_classifier.fit(tfidf_train_vectors, labels)
        dump(one_vs_rest_classifier, self.get_model_path())

    def predict(self, predictions_samples: list[PredictionSample]) -> list[list[Option]]:
        train_texts = load(self.get_data_path())

        vectorized = TfidfVectorizer()
        vectorized.fit_transform(train_texts)

        predict_texts = [sample.get_input_text() for sample in predictions_samples]

        tfidf_predict_vectors = vectorized.transform(predict_texts)

        classifier = load(self.get_model_path())
        predictions_text = classifier.predict(tfidf_predict_vectors)
        predictions_scores = [prediction for prediction in predictions_text.tolist()]
        return self.predictions_to_options_list(predictions_scores)
