import os
from functools import lru_cache
from os.path import join, exists
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from joblib import dump, load
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.MultiLabelMethod import MultiLabelMethod
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData

nltk.download("wordnet")
nltk.download("stopwords")

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words("english")
stop_words_set = set(stop_words)

lemmatize = lru_cache(maxsize=50000)(lemmatizer.lemmatize)


class TfIdfMethod(MultiLabelMethod):

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        return True

    def get_data_path(self):
        model_folder_path = self.get_path()

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        return join(model_folder_path, "data.txt")

    def get_model_path(self):
        model_folder_path = self.get_path()

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        return join(model_folder_path, "fast.model")

    def train(self, multi_option_data: ExtractionData):
        texts = [sample.pdf_data.get_text() for sample in multi_option_data.samples]
        dump(texts, self.get_data_path())

        vectorized = TfidfVectorizer()
        tfidf_train_vectors = vectorized.fit_transform(texts)

        labels = self.get_one_hot_encoding(multi_option_data)
        one_vs_rest_classifier = OneVsRestClassifier(RandomForestClassifier())
        one_vs_rest_classifier = one_vs_rest_classifier.fit(tfidf_train_vectors, labels)
        dump(one_vs_rest_classifier, self.get_model_path())

    def predict(self, prediction_samples_data: PredictionSamplesData) -> list[list[Value]]:
        texts = [sample.pdf_data.get_text() for sample in prediction_samples_data.prediction_samples]
        texts = [text.replace("\n", " ") for text in texts]

        model = self.load_model()
        predictions = model.predict(texts)

        if prediction_samples_data.multi_value:
            predictions_proba = model.predict_proba(texts)
            threshold = 0.5
            predictions = (predictions_proba > threshold).astype(int)

        predictions_values = list()
        for prediction in predictions:
            if prediction_samples_data.multi_value:
                prediction_indices = [i for i, value in enumerate(prediction) if value == 1]
            else:
                prediction_indices = [prediction] if isinstance(prediction, int) else prediction.tolist()

            sample_predictions = list()
            for prediction_index in prediction_indices:
                if 0 <= prediction_index < len(prediction_samples_data.options):
                    sample_predictions.append(Value.from_option(prediction_samples_data.options[prediction_index]))

            predictions_values.append(sample_predictions)

        return predictions_values
