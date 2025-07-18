import os
import shutil
from math import exp
from os.path import join, exists
import evaluate
import numpy as np
import pandas as pd
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
)
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.use_cases.extractors.bert_method_scripts.AvoidEvaluation import AvoidEvaluation
from trainable_entity_extractor.use_cases.extractors.bert_method_scripts.EarlyStoppingAfterInitialTraining import (
    EarlyStoppingAfterInitialTraining,
)
from trainable_entity_extractor.use_cases.extractors.bert_method_scripts.get_batch_size import get_batch_size, get_max_steps
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.MultiLabelMethod import MultiLabelMethod
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample

MODEL_NAME = "google-bert/bert-base-uncased"

clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class BertSeqSteps(MultiLabelMethod):

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

    @staticmethod
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = 1 / (1 + np.exp(-predictions))
        predictions = (predictions > 0.5).astype(int).reshape(-1)
        return clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))

    def preprocess_function(self, sample: TrainingSample):
        text = sample.get_segments_text()
        labels = [1.0 if value in sample.labeled_data.values else 0.0 for value in self.options]

        example = tokenizer(text, padding="max_length", truncation="only_first", max_length=self.get_token_length())
        example["labels"] = labels
        return example

    def train(self, multi_option_data: ExtractionData):
        shutil.rmtree(self.get_model_path(), ignore_errors=True)

        self.create_dataset(multi_option_data, "train")

        examples = [self.preprocess_function(x) for x in multi_option_data.samples]
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        id2class = {index: label for index, label in enumerate([x.label for x in self.options])}
        class2id = {label: index for index, label in enumerate([x.label for x in self.options])}

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(self.options),
            id2label=id2class,
            label2id=class2id,
            problem_type="multi_label_classification",
        )

        training_args = TrainingArguments(
            output_dir=self.get_model_path(),
            learning_rate=2e-5,
            per_device_train_batch_size=get_batch_size(len(multi_option_data.samples)),
            per_device_eval_batch_size=get_batch_size(len(multi_option_data.samples)),
            max_steps=get_max_steps(len(multi_option_data.samples)),
            weight_decay=0.01,
            eval_steps=200,
            save_steps=200,
            save_strategy="steps",
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            fp16=False,
            bf16=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=examples,
            eval_dataset=examples,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingAfterInitialTraining(early_stopping_patience=3), AvoidEvaluation()],
        )

        trainer.train()

        trainer.save_model(self.get_model_path())

    @staticmethod
    def logit_to_probabilities(logits):
        odds = [1 / (1 + exp(-logit)) for logit in logits]
        return odds

    def predict(self, multi_option_data: ExtractionData) -> list[list[Value]]:
        id2class = {index: label for index, label in enumerate([x.label for x in self.options])}
        class2id = {label: index for index, label in enumerate([x.label for x in self.options])}

        self.create_dataset(multi_option_data, "predict")

        model = AutoModelForSequenceClassification.from_pretrained(
            self.get_model_path(),
            num_labels=len(self.options),
            id2label=id2class,
            label2id=class2id,
            problem_type="multi_label_classification",
        )

        model.eval()

        inputs = tokenizer(
            [x.pdf_data.get_text() for x in multi_option_data.samples],
            return_tensors="pt",
            padding="max_length",
            truncation="only_first",
            max_length=self.get_token_length(),
        )
        output = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])

        return self.predictions_to_options_list([self.logit_to_probabilities(logit) for logit in output.logits])

    def get_token_length(self):
        data = pd.read_csv(self.get_data_path("train"))
        max_length = 0
        for index, row in data.iterrows():
            length = len(tokenizer(row["text"]).data["input_ids"])
            max_length = max(length, max_length)

        return max_length
