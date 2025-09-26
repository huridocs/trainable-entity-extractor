import json
from collections import Counter
from pathlib import Path
from time import time

import nltk
import numpy as np

import lightgbm as lgb
from sklearn.metrics import f1_score

from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.adapters.extractors.segment_selector.methods.avoiding_words.SegmentAvoidingWords import (
    SegmentAvoidingWords,
)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("stopwords")


class AvoidingWords:
    def __init__(self):
        self.segments: list[SegmentAvoidingWords] = list()
        self.model = None
        self.best_cut = 0

    def create_model(self, training_pdfs_segments: list[PdfData], model_path):
        start = time()
        self.set_segments(pdfs_segments=training_pdfs_segments)

        config_logger.info(f"Set segments {int(time() - start)} seconds")

        if len(self.segments) == 0:
            return None

        self.save_most_frequent_words(model_path)
        self.set_most_frequent_words_to_segments(model_path)
        start = time()
        self.run_light_gbm()
        config_logger.info(f"Run lightGBM {int(time() - start)} seconds")

        return self.model

    @staticmethod
    def set_model(x_train, y_train):
        parameters = dict()
        parameters["num_leaves"] = 70
        parameters["feature_fraction"] = 0.9
        parameters["bagging_fraction"] = 0.9
        parameters["bagging_freq"] = 0
        parameters["objective"] = "binary"
        parameters["learning_rate"] = 0.05
        parameters["metric"] = "binary_logloss"
        parameters["verbose"] = -1
        parameters["boosting_type"] = "gbdt"

        train_data = lgb.Dataset(x_train, y_train)
        num_round = 3000
        light_gbm_model = lgb.train(parameters, train_data, num_round)

        return light_gbm_model

    def run_light_gbm(self):
        x_train, y_train = self.get_training_data()
        self.model = self.set_model(x_train, y_train)

    def get_training_data(self):
        y = np.array([])
        x_rows = []
        for segment in self.segments:
            x_rows.append(segment.get_features_array())
            y = np.append(y, segment.pdf_segment.ml_label)

        X = np.zeros((len(x_rows), len(x_rows[0]) if x_rows else 0))
        for i, v in enumerate(x_rows):
            X[i] = v

        return X, y

    def set_segments(self, pdfs_segments: list[PdfData]):
        self.segments = list()
        for pdf_features in pdfs_segments:
            self.segments.extend(SegmentAvoidingWords.from_pdf_features(pdf_features))

    def predict(self, model, testing_pdfs_segments: list[PdfData], model_path):
        self.set_segments(testing_pdfs_segments)
        self.set_most_frequent_words_to_segments(model_path)
        x, y = self.get_training_data()
        x = x[:, : model.num_feature()]
        predictions = model.predict(x)
        return predictions

    @staticmethod
    def get_performance(predictions, y_truth):
        return f1_score(y_truth, [prediction > 0.5 for prediction in predictions])

    def save_most_frequent_words(self, model_path):
        appearing_words = Counter()
        for segment in [segment for segment in self.segments if segment.pdf_segment.ml_label]:
            text_tokens = word_tokenize(segment.text_content)
            appearing_words.update(
                [word.lower() for word in text_tokens if word.lower() not in stopwords.words() and word not in ".,"]
            )

        most_repeated_words = [x[0] for x in appearing_words.most_common()[:3]]

        avoiding_words = Counter()
        for segment in [segment for segment in self.segments if segment.pdf_segment.ml_label]:
            text_tokens = word_tokenize(segment.next_segment.text_content) if segment.next_segment else []
            text_tokens += word_tokenize(segment.previous_segment.text_content) if segment.previous_segment else []
            avoiding_words.update(
                [
                    word.lower()
                    for word in text_tokens
                    if word.lower() not in stopwords.words() and word not in ".," and word.lower() not in appearing_words
                ]
            )

        config_logger.info(avoiding_words.most_common()[:4])
        most_repeated_words.extend([x[0] for x in avoiding_words.most_common()[:4]])
        self.get_frequent_words_path(model_path).write_text(json.dumps(most_repeated_words))

    @staticmethod
    def get_frequent_words_path(model_path: str | Path) -> Path:
        return Path(str(model_path) + "_frequent_words")

    def set_most_frequent_words_to_segments(self, model_path: str):
        most_frequent_words = json.loads(Path(self.get_frequent_words_path(model_path)).read_text())
        for segment in self.segments:
            segment.set_most_frequent_words(most_frequent_words)
