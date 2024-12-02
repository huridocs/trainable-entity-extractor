import os
import shutil
from os.path import join, exists

import pandas as pd
from datasets import load_dataset, DatasetDict
from fastfit import FastFitTrainer, FastFit, sample_dataset
from transformers import AutoTokenizer, pipeline

from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.Option import Option
from setfit import SetFitModel, TrainingArguments, Trainer

from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.extractors.ExtractorBase import ExtractorBase
from trainable_entity_extractor.extractors.bert_method_scripts.AvoidAllEvaluation import AvoidAllEvaluation
from trainable_entity_extractor.extractors.bert_method_scripts.EarlyStoppingAfterInitialTraining import (
    EarlyStoppingAfterInitialTraining,
)
from trainable_entity_extractor.extractors.bert_method_scripts.get_batch_size import get_batch_size, get_max_steps
from trainable_entity_extractor.extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import (
    TextToMultiOptionMethod,
)


class TextFastFit(TextToMultiOptionMethod):

    model_name = "sentence-transformers/paraphrase-mpnet-base-v2"

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        if extraction_data.multi_value:
            return False

        return True

    def get_data_path(self):
        model_folder_path = join(self.extraction_identifier.get_path(), self.get_name())

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        return join(model_folder_path, "data.csv")

    def get_model_path(self):
        model_folder_path = join(self.extraction_identifier.get_path(), self.get_name())

        if not exists(model_folder_path):
            os.makedirs(model_folder_path)

        model_path = join(model_folder_path, "fast_fit_model")

        os.makedirs(model_path, exist_ok=True)

        return str(model_path)

    @staticmethod
    def eval_encodings(example):
        example["label"] = eval(example["label"])
        return example

    def get_dataset_from_data(self, extraction_data: ExtractionData):
        data = list()
        texts = [self.get_text(sample.labeled_data.source_text) for sample in extraction_data.samples]
        labels = list()

        for sample in extraction_data.samples:
            labels.append("no_label")
            if sample.labeled_data.values:
                labels[-1] = self.options[self.options.index(sample.labeled_data.values[0])].label

        for text, label in zip(texts[:10000], labels[:10000]):
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
        dataset_dict = DatasetDict()
        dataset_dict["train"] = train_dataset
        dataset_dict["validation"] = train_dataset
        dataset_dict["test"] = train_dataset
        trainer = FastFitTrainer(
            model_name_or_path="sentence-transformers/paraphrase-mpnet-base-v2",
            label_column_name="label",
            text_column_name="text",
            output_dir=self.get_model_path(),
            max_steps=get_max_steps(len(extraction_data.samples)),
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=20000,
            save_steps=20000,
            load_best_model_at_end=True,
            max_text_length=128,
            dataloader_drop_last=False,
            num_repeats=4,
            optim="adafactor",
            clf_loss_factor=0.1,
            fp16=True,
            dataset=dataset_dict,
        )

        model = trainer.train()
        model.save_pretrained(self.get_model_path())

    def predict(self, predictions_samples: list[PredictionSample]) -> list[list[Option]]:
        model = FastFit.from_pretrained(self.get_model_path())
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

        texts = [self.get_text(sample.source_text) for sample in predictions_samples]
        predictions = list()
        for text in texts:
            prediction = classifier(text)
            prediction_labels = [x["label"] for x in prediction if x["score"] > 0.5]
            predictions.append([x for x in self.options if x.label in prediction_labels])

        return predictions
