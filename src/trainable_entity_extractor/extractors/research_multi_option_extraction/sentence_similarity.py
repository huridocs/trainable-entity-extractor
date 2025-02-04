import random
from os.path import join
from time import time

import numpy as np
from fast_trainer.PdfSegment import PdfSegment
from pdf_token_type_labels.TokenType import TokenType
from sentence_transformers.util import cos_sim

from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score

from trainable_entity_extractor.config import ROOT_PATH
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.PdfLabels import PdfLabels
from pdf_topic_classification.pdf_topic_classification_data import get_labeled_data
import lightgbm as lgb

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
valid_types = [TokenType.TEXT, TokenType.LIST_ITEM, TokenType.TITLE, TokenType.SECTION_HEADER, TokenType.CAPTION]


def one_hot_to_options_list(options, pdfs_options: list[list[str]]) -> list[list[int]]:
    options_one_hot: list[list[int]] = list()
    for pdf_options in pdfs_options:
        pdf_options_one_hot = [0] * len(options)

        for pdf_option in pdf_options:
            if pdf_option in options:
                pdf_options_one_hot[options.index(pdf_option)] = 1

        options_one_hot.append(pdf_options_one_hot)

    return options_one_hot


def get_similarity(sentence_1: str, sentence_2: str):
    embeddings1 = model.encode(sentence_1, convert_to_tensor=True)
    embeddings2 = model.encode(sentence_2, convert_to_tensor=True)
    return float(cos_sim(embeddings1, embeddings2))


def get_five_values(values):
    return values[:5] + [0] * (5 - len(values))


def get_features(option, pdfs_labels: list[PdfLabels]):
    features = list()
    for pdf_labels in pdfs_labels:
        pdf_segments = [PdfSegment.from_pdf_tokens(paragraph.tokens) for paragraph in pdf_labels.paragraphs]
        pdf_segments = [x for x in pdf_segments if x.page_number < 4]
        texts_types = [
            (pdf_segment.text_cleaned, pdf_segment.paragraph_type)
            for pdf_segment in pdf_segments
            if pdf_segment.paragraph_type in valid_types
        ]
        similarities_types = [(get_similarity(option, text), type) for text, type in texts_types]
        similarities_titles = [score for score, sentence_type in similarities_types if sentence_type == TokenType.TITLE]
        similarities_no_titles = [score for score, sentence_type in similarities_types if sentence_type != TokenType.TITLE]
        maximum = max(similarities_types, key=lambda x: x[0]) if similarities_types else (0, TokenType.TEXT)
        one_hot_maximum = [1 if index == maximum[1].get_index() else 0 for index in range(len(TokenType))]
        scores = [score for score, _ in similarities_types]
        average = sum(scores) / len(scores) if scores else 0

        features.append(
            one_hot_maximum
            + [maximum[0], average]
            + get_five_values(similarities_titles)
            + get_five_values(similarities_no_titles)
        )

    return features


def get_model_name(option: str):
    return join(ROOT_PATH, "data", f"{option.replace(' ', '')}.model")


def train_models(task_labeled_data, train_set: list[PdfLabels]):
    for option in task_labeled_data.values:
        labels = [1 if option in x.labels else 0 for x in train_set]
        features = get_features(option, train_set)
        lgb_train = lgb.Dataset(np.array(features), labels)
        lgb_eval = lgb.Dataset(np.array(features), labels, reference=lgb_train)

        params = {
            "boosting_type": "gbdt",
            "objective": "regression",
        }

        gbm = lgb.train(
            params, lgb_train, num_boost_round=20, valid_sets=[lgb_eval], callbacks=[lgb.early_stopping(stopping_rounds=5)]
        )

        gbm.save_model(get_model_name(option))


def get_train_test(task_labeled_data) -> (list[PdfLabels], list[PdfLabels]):
    train_size = int(len(task_labeled_data.pdfs_labels) * 0.8)
    random.seed(22)
    train_set: list[PdfLabels] = random.choices(task_labeled_data.pdfs_labels, k=train_size)
    test_set: list[PdfLabels] = [x for x in task_labeled_data.pdfs_labels if x not in train_set]

    train_set = [x for x in train_set if len(x.paragraphs)]

    return train_set, test_set


def get_predictions(task_labeled_data, test_set: list[PdfLabels]):
    truth_one_hot = one_hot_to_options_list(task_labeled_data.values, [x.labels for x in test_set])
    predictions = list()
    for test_pdf in test_set[:5]:
        pdf_labels = list()
        if not test_pdf.paragraphs:
            predictions.append(pdf_labels)
            continue

        for option in task_labeled_data.values:
            gbm = lgb.Booster(model_file=get_model_name(option))
            features = get_features(option, [test_pdf])
            y_pred = gbm.predict(features)
            if y_pred[0] > 0.5:
                pdf_labels.append(option)

        predictions.append(pdf_labels)

    predictions_one_hot = one_hot_to_options_list(task_labeled_data.values, predictions)
    print(100 * f1_score(truth_one_hot, predictions_one_hot, average="macro"))


def run():
    task_labeled_data = get_labeled_data("cyrilla")[0]

    train_set, test_set = get_train_test(task_labeled_data)

    # train_models(task_labeled_data, train_set)
    get_predictions(task_labeled_data, test_set)


if __name__ == "__main__":
    start = time()
    print("start")
    # run()
    print(
        get_similarity(
            """“electronic communication” means any form of communication transmitted
or communicated electronically and includes any text message, writing,
photograph, picture, recording or other matter that is communicated
electronically;""",
            "Freedom of Expression",
        )
    )
    print(
        get_similarity(
            """“electronic communication” means any form of communication transmitted
or communicated electronically and includes any text message, writing,
photograph, picture, recording or other matter that is communicated
electronically;""",
            "Privacy",
        )
    )
    print("finished in", round(time() - start, 1), "seconds")
