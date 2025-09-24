import os
import shutil
from os.path import join, exists
import pandas as pd
import torch
from datasets import load_dataset
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from setfit import SetFitModel, TrainingArguments, Trainer

from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.ports.ExtractorBase import ExtractorBase
from trainable_entity_extractor.adapters.extractors.bert_method_scripts.AvoidAllEvaluation import AvoidAllEvaluation
from trainable_entity_extractor.adapters.extractors.bert_method_scripts.EarlyStoppingAfterInitialTraining import (
    EarlyStoppingAfterInitialTraining,
)
from trainable_entity_extractor.adapters.extractors.bert_method_scripts.get_batch_size import get_batch_size, get_max_steps
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.MultiLabelMethod import MultiLabelMethod
import gc


class SetFitEnglishMethod(MultiLabelMethod):
    model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

    def gpu_needed(self) -> bool:
        return True

    def get_data_path(self):
        model_folder_path = self.get_path()

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        return join(model_folder_path, "data.csv")

    def get_model_path(self):
        model_folder_path = self.get_path()

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        model_path = join(model_folder_path, "setfit_model")

        os.makedirs(model_path, exist_ok=True)

        return str(model_path)

    @staticmethod
    def eval_encodings(example):
        example["label"] = eval(example["label"])
        return example

    def get_dataset_from_data(self, extraction_data: ExtractionData):
        data = list()
        texts = [sample.pdf_data.get_text() for sample in extraction_data.samples]
        labels = self.get_one_hot_encoding(extraction_data)

        for text, label in zip(texts, labels):
            data.append([text, label])

        df = pd.DataFrame(data)
        df.columns = ["text", "label"]

        df.to_csv(self.get_data_path())
        dataset_csv = load_dataset("csv", data_files=self.get_data_path())
        dataset = dataset_csv["train"]
        dataset = dataset.map(self.eval_encodings)

        return dataset

    def train(self, extraction_data: ExtractionData):
        shutil.rmtree(self.get_model_path(), ignore_errors=True)
        train_dataset = self.get_dataset_from_data(extraction_data)
        batch_size = get_batch_size(len(extraction_data.samples))

        model = SetFitModel.from_pretrained(
            self.model_name,
            labels=[x.label for x in self.options],
            multi_target_strategy="one-vs-rest",
            trust_remote_code=True,
        )

        args = TrainingArguments(
            output_dir=self.get_model_path(),
            batch_size=batch_size,
            max_steps=get_max_steps(len(extraction_data.samples)),
            save_strategy="steps",
            eval_steps=200,
            save_steps=200,
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            metric="f1",
            callbacks=[EarlyStoppingAfterInitialTraining(early_stopping_patience=3), AvoidAllEvaluation()],
        )

        trainer.train()

        trainer.model.save_pretrained(self.get_model_path())

        del model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    def predict(self, prediction_samples_data: PredictionSamplesData) -> list[list[Value]]:
        texts = [sample.pdf_data.get_text() for sample in prediction_samples_data.prediction_samples]
        texts = [text.replace("\n", " ") for text in texts]

        model = SetFitModel.from_pretrained(self.get_model_path())
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

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        if not torch.cuda.is_available():
            return False

        if not extraction_data.multi_value:
            return False

        if ExtractorBase.is_multilingual(extraction_data):
            return False

        return True
