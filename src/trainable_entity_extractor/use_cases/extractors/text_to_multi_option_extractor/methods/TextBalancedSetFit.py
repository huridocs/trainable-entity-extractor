import os
import shutil
from os.path import join, exists

import pandas as pd
from datasets import load_dataset

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.Option import Option
from setfit import SetFitModel, TrainingArguments, Trainer

from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.use_cases.extractors.ExtractorBase import ExtractorBase
from trainable_entity_extractor.use_cases.extractors.bert_method_scripts.AvoidAllEvaluation import AvoidAllEvaluation
from trainable_entity_extractor.use_cases.extractors.bert_method_scripts.EarlyStoppingAfterInitialTraining import (
    EarlyStoppingAfterInitialTraining,
)
from trainable_entity_extractor.use_cases.extractors.bert_method_scripts.get_batch_size import get_batch_size, get_max_steps
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import (
    TextToMultiOptionMethod,
)


class TextBalancedSetFit(TextToMultiOptionMethod):

    model_name = "sentence-transformers/paraphrase-mpnet-base-v2"
    MAX_SAMPLES = 500

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        if not extraction_data.multi_value:
            return False

        if not ExtractorBase.is_multilingual(extraction_data):
            return True

        return False

    def get_data_path(self):
        model_folder_path = join(self.extraction_identifier.get_path(), self.get_name())

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        return join(model_folder_path, "data.csv")

    def get_model_path(self):
        model_folder_path = join(self.extraction_identifier.get_path(), self.get_name())

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        model_path = join(model_folder_path, "setfit_model")

        os.makedirs(model_path, exist_ok=True)

        return str(model_path)

    @staticmethod
    def eval_encodings(example):
        example["label"] = eval(example["label"])
        return example

    def get_balanced_data(self, data: list[tuple[str, list[int]]]):
        balanced_data = list()
        rows_added = list()
        columns_counts = {i: 0 for i in range(len(data[0][1]))}

        self.add_rows_to_balanced_data(balanced_data, columns_counts, data, rows_added, self.MAX_SAMPLES / 5)
        self.add_rows_to_balanced_data(balanced_data, columns_counts, data, rows_added, self.MAX_SAMPLES / 2)
        self.add_rows_to_balanced_data(balanced_data, columns_counts, data, rows_added, self.MAX_SAMPLES)

        return balanced_data

    @staticmethod
    def add_rows_to_balanced_data(balanced_data, columns_counts, data, rows_added, max_samples):
        for row_index, (text, labels) in enumerate(data):
            if row_index in rows_added:
                continue

            for i, label in enumerate(labels):
                if not label:
                    continue

                if columns_counts[i] >= max_samples:
                    continue

                for k in range(len(labels)):
                    if labels[k]:
                        columns_counts[k] += 1

                rows_added.append(row_index)
                balanced_data.append((text, labels))
                break

    def get_dataset_from_data(self, extraction_data: ExtractionData):
        data = list()
        texts = [self.get_text(sample.labeled_data.source_text) for sample in extraction_data.samples]
        labels = self.get_one_hot_encoding(extraction_data)

        for text, label in zip(texts, labels):
            data.append((text, label))

        balance_data = self.get_balanced_data(data)
        df = pd.DataFrame(balance_data)
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
        )

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
            metric="f1",
            callbacks=[EarlyStoppingAfterInitialTraining(early_stopping_patience=3), AvoidAllEvaluation()],
        )

        trainer.train()

        trainer.model.save_pretrained(self.get_model_path())

    def predict(self, predictions_samples: list[PredictionSample]) -> list[list[Option]]:
        model = SetFitModel.from_pretrained(self.get_model_path())
        texts = [self.get_text(sample.source_text) for sample in predictions_samples]
        predictions = model.predict(texts)

        return self.predictions_to_options_list(predictions.tolist())
