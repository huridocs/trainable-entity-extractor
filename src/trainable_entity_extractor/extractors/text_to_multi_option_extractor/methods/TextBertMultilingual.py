import os
import shutil
from math import exp
from os.path import join, exists

import pandas as pd
from transformers import TrainingArguments, AutoTokenizer

from trainable_entity_extractor.data.Option import Option
from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.extractors.ExtractorBase import ExtractorBase
from trainable_entity_extractor.extractors.bert_method_scripts.get_batch_size import get_batch_size

from trainable_entity_extractor.extractors.bert_method_scripts.multi_label_sequence_classification_trainer import (
    multi_label_run,
    MultiLabelDataTrainingArguments,
    ModelArguments,
)
from trainable_entity_extractor.extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import (
    TextToMultiOptionMethod,
)

MODEL_NAME = "google-bert/bert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class TextBertMultilingual(TextToMultiOptionMethod):
    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        if not extraction_data.multi_value:
            return False

        if ExtractorBase.is_multilingual(extraction_data):
            return True

        return False

    def get_data_path(self, name):
        model_folder_path = join(self.extraction_identifier.get_path(), self.get_name())

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        return join(model_folder_path, f"{name}.csv")

    def get_model_path(self):
        model_folder_path = join(self.extraction_identifier.get_path(), self.get_name())

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        model_path = join(model_folder_path, "model")

        os.makedirs(model_path, exist_ok=True)

        return str(model_path)

    @staticmethod
    def get_text(texts: list[str]):
        words = list()
        for text in texts:
            text_words = text.split()
            for word in text_words:
                clean_word = "".join([x for x in word if x.isalpha() or x.isdigit()])

                if clean_word:
                    words.append(clean_word)

        return " ".join(words)

    def create_dataset(self, multi_option_data: ExtractionData, name: str):
        texts = [self.get_text(sample.tags_texts) for sample in multi_option_data.samples]
        labels = self.get_one_hot_encoding(multi_option_data)
        return self.save_dataset(texts, labels, name)

    def save_dataset(self, texts, labels, name):
        rows = list()

        for text, label in zip(texts, labels):
            rows.append([text, label])

        output_df = pd.DataFrame(rows)
        output_df.columns = ["text", "labels"]

        if name != "predict":
            output_df = output_df.sample(frac=1, random_state=22).reset_index(drop=True)

        output_df.to_csv(str(self.get_data_path(name)))
        return self.get_data_path(name)

    def train(self, multi_option_data: ExtractionData):
        shutil.rmtree(self.get_model_path(), ignore_errors=True)

        training_csv_path = self.create_dataset(multi_option_data, "train")
        validation_csv_path = self.create_dataset(multi_option_data, "validation")
        model_arguments = ModelArguments(MODEL_NAME)
        labels_number = len(self.options)

        data_training_arguments = MultiLabelDataTrainingArguments(
            train_file=training_csv_path,
            validation_file=validation_csv_path,
            max_seq_length=256,
            labels_number=labels_number,
        )

        batch_size = get_batch_size(len(multi_option_data.samples))
        t5_training_arguments = TrainingArguments(
            report_to=[],
            output_dir=self.get_model_path(),
            overwrite_output_dir=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=batch_size,
            eval_accumulation_steps=batch_size,
            learning_rate=5e-05,
            do_train=True,
            do_eval=False,
            do_predict=False,
            save_total_limit=2,
            save_strategy="steps",
            evaluation_strategy="steps",
            eval_steps=500000,
            save_steps=200,
            load_best_model_at_end=False,
            logging_steps=50,
            metric_for_best_model="f1",
            num_train_epochs=23,
        )

        multi_label_run(model_arguments, data_training_arguments, t5_training_arguments)

    @staticmethod
    def logit_to_probabilities(logits):
        odds = [1 / (1 + exp(-logit)) for logit in logits]
        return odds

    def predict(self, predictions_samples: list[PredictionSample]) -> list[list[Option]]:
        labels_number = len(self.options)

        texts = [self.get_text(sample.tags_texts) for sample in predictions_samples]
        labels = [[0] * len(self.options) for _ in predictions_samples]

        predict_path = self.save_dataset(texts, labels, "predict")
        model_arguments = ModelArguments(self.get_model_path(), ignore_mismatched_sizes=True)
        data_training_arguments = MultiLabelDataTrainingArguments(
            train_file=predict_path,
            validation_file=predict_path,
            test_file=predict_path,
            max_seq_length=256,
            labels_number=labels_number,
        )

        batch_size = get_batch_size(len(predictions_samples))
        t5_training_arguments = TrainingArguments(
            report_to=[],
            output_dir=self.get_model_path(),
            overwrite_output_dir=False,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=batch_size,
            eval_accumulation_steps=batch_size,
            do_train=False,
            do_eval=False,
            do_predict=True,
        )

        logits = multi_label_run(model_arguments, data_training_arguments, t5_training_arguments)
        return self.predictions_to_options_list([self.logit_to_probabilities(logit) for logit in logits])
