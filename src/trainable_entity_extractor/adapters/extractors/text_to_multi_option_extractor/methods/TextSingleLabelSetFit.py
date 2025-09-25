import os
import shutil
from os.path import join, exists

import pandas as pd
from datasets import load_dataset

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.Option import Option
from setfit import SetFitModel, TrainingArguments, Trainer

from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.ports.ExtractorBase import ExtractorBase
from trainable_entity_extractor.adapters.extractors.bert_method_scripts.AvoidAllEvaluation import AvoidAllEvaluation
from trainable_entity_extractor.adapters.extractors.bert_method_scripts.get_batch_size import get_batch_size, get_max_steps
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import (
    TextToMultiOptionMethod,
)


class TextSingleLabelSetFit(TextToMultiOptionMethod):
    model_name = "sentence-transformers/paraphrase-mpnet-base-v2"

    def gpu_needed(self) -> bool:
        return True

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        if extraction_data.multi_value:
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

        model_path = join(model_folder_path, "single_setfit_model")

        os.makedirs(model_path, exist_ok=True)

        return str(model_path)

    @staticmethod
    def eval_encodings(example):
        example["label"] = eval(example["label"])
        return example

    def get_dataset_from_data(self, extraction_data: ExtractionData):
        data = list()
        texts = [self.get_text(sample.get_input_text()) for sample in extraction_data.samples]
        labels = list()

        for sample in extraction_data.samples:
            labels.append("no_label")
            if sample.labeled_data.values:
                labels[-1] = self.options[self.options.index(sample.labeled_data.values[0])].label

        for text, label in zip(texts[:15000], labels[:15000]):
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

        model = SetFitModel.from_pretrained(self.model_name, labels=[x.label for x in self.options])

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
            callbacks=[AvoidAllEvaluation()],
        )

        trainer.train()

        trainer.model.save_pretrained(self.get_model_path())

    def predict(self, prediction_samples: PredictionSamplesData) -> list[list[Option]]:
        self.options = prediction_samples.options
        self.multi_value = prediction_samples.multi_value
        model = SetFitModel.from_pretrained(self.get_model_path())
        texts = [self.get_text(sample.get_input_text()) for sample in prediction_samples.prediction_samples]
        predictions = model.predict(texts)

        return [
            [option for option in prediction_samples.options if option.label == prediction] for prediction in predictions
        ]
