import os
import shutil
from math import exp
from os.path import join, exists

import pandas as pd
from transformers import TrainingArguments, AutoTokenizer

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.adapters.extractors.bert_method_scripts.get_batch_size import get_batch_size
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.MultiLabelMethod import MultiLabelMethod

from trainable_entity_extractor.adapters.extractors.bert_method_scripts.multi_label_sequence_classification_trainer import (
    multi_label_run,
    MultiLabelDataTrainingArguments,
    ModelArguments,
)

MODEL_NAME = "google-bert/bert-base-uncased"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class BertMethod(MultiLabelMethod):
    def gpu_needed(self) -> bool:
        return True

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        return extraction_data.multi_value

    def get_data_path(self, name):
        model_folder_path = self.get_path()

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        return join(model_folder_path, f"{name}.csv")

    def get_model_path(self):
        model_folder_path = self.get_path()

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        model_path = join(model_folder_path, "model")

        os.makedirs(model_path, exist_ok=True)

        return str(model_path)

    def create_dataset(self, multi_option_data: ExtractionData, name: str):
        texts, labels = self.get_texts_labels(multi_option_data)

        rows = list()

        for text, label in zip(texts, labels):
            rows.append([text, label])

        output_df = pd.DataFrame(rows)
        output_df.columns = ["text", "labels"]

        if name != "predict":
            output_df = output_df.sample(frac=1, random_state=22).reset_index(drop=True)

        output_df.to_csv(self.get_data_path(name))
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
