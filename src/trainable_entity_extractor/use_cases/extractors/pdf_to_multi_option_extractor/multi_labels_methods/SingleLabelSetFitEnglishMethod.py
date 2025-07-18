import gc
import os
import shutil
from os.path import join, exists

import pandas as pd
import torch.cuda
from datasets import load_dataset

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.Option import Option
from setfit import SetFitModel, TrainingArguments, Trainer

from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.use_cases.extractors.ExtractorBase import ExtractorBase
from trainable_entity_extractor.use_cases.extractors.bert_method_scripts.AvoidAllEvaluation import AvoidAllEvaluation
from trainable_entity_extractor.use_cases.extractors.bert_method_scripts.EarlyStoppingAfterInitialTraining import (
    EarlyStoppingAfterInitialTraining,
)
from trainable_entity_extractor.use_cases.extractors.bert_method_scripts.get_batch_size import get_batch_size, get_max_steps
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.MultiLabelMethod import MultiLabelMethod
from trainable_entity_extractor.use_cases.send_logs import send_logs


class SingleLabelSetFitEnglishMethod(MultiLabelMethod):

    model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        if not torch.cuda.is_available():
            send_logs(self.extraction_identifier, f"GPU not available for {self.get_name()}")
            return False

        if extraction_data.multi_value:
            return False

        if ExtractorBase.is_multilingual(extraction_data):
            return False

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
        labels = list()

        for sample in extraction_data.samples:
            labels.append("no_label")
            if sample.labeled_data.values:
                options = [option for option in self.options if option.id == sample.labeled_data.values[0].id]
                if options:
                    labels[-1] = options[0].label

        for text, label in zip(texts, labels):
            data.append([text, label])

        df = pd.DataFrame(data)
        df.columns = ["text", "label"]

        df.to_csv(self.get_data_path())
        dataset_csv = load_dataset("csv", data_files=self.get_data_path())
        dataset = dataset_csv["train"]

        return dataset

    def train(self, extraction_data: ExtractionData):
        shutil.rmtree(self.get_model_path(), ignore_errors=True)
        train_dataset = self.get_dataset_from_data(extraction_data)
        batch_size = get_batch_size(len(extraction_data.samples))

        model = SetFitModel.from_pretrained(self.model_name, labels=[x.label for x in self.options], trust_remote_code=True)

        args = TrainingArguments(
            output_dir=self.get_model_path(),
            batch_size=batch_size,
            max_steps=get_max_steps(len(extraction_data.samples)),
            evaluation_strategy="steps",
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
            metric="accuracy",
            callbacks=[EarlyStoppingAfterInitialTraining(early_stopping_patience=3), AvoidAllEvaluation()],
        )

        trainer.train()

        trainer.model.save_pretrained(self.get_model_path())

        del model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    def predict(self, multi_option_data: ExtractionData) -> list[list[Value]]:
        model = SetFitModel.from_pretrained(self.get_model_path(), trust_remote_code=True)
        predict_texts = [sample.pdf_data.get_text() for sample in multi_option_data.samples]
        predictions = model.predict(predict_texts)

        del model
        gc.collect()
        torch.cuda.empty_cache()

        return [
            [Value.from_option(option) for option in self.options if option.label == prediction]
            for prediction in predictions
        ]
