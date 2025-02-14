import logging
import os
import shutil
from functools import lru_cache
from os.path import join, exists, dirname
from pathlib import Path

import torch
import transformers
from huggingface_hub import hf_hub_download
import pandas as pd
import csv
from transformers.utils import logging as logging_hf
from transformers import AutoTokenizer, MT5ForConditionalGeneration

from trainable_entity_extractor.config import DATA_PATH, config_logger
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.extractors.ToTextExtractorMethod import ToTextExtractorMethod

from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.TrueCaser import TrueCaser
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.run_seq_2_seq import (
    ModelArguments,
    DataTrainingArguments,
    run,
    T5TrainingArguments,
)

transformers.logging.set_verbosity_error()
logging.getLogger("pytorch_pretrained_bert.tokenization").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

logging_hf.set_verbosity(40)


class MT5TrueCaseEnglishSpanishMethod(ToTextExtractorMethod):
    SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

    def get_model_path(self):
        return str(join(self.get_path(), "model"))

    def get_max_length_path(self):
        return Path(self.get_path(), "max_length_output")

    def get_max_input_length(self, extraction_data: ExtractionData):
        tokenizer = AutoTokenizer.from_pretrained("HURIDOCS/mt5-small-spanish-es", cache_dir=self.get_cache_dir())
        texts = [
            self.extraction_identifier.run_name + ": " + " ".join(x.segment_selector_texts) for x in extraction_data.samples
        ]
        tokens_number = [len(tokenizer(text)["input_ids"]) for text in texts]
        input_length = min(int((max(tokens_number) + 5) * 1.5), 512)
        config_logger.info(f"Max input length: {str(input_length)}")
        return input_length

    def get_max_output_length(self, extraction_data: ExtractionData):
        tokenizer = AutoTokenizer.from_pretrained("HURIDOCS/mt5-small-spanish-es", cache_dir=self.get_cache_dir())
        tokens_number = [len(tokenizer(" ".join(x.segment_selector_texts))["input_ids"]) for x in extraction_data.samples]
        output_length = min(int((max(tokens_number) + 5) * 1.5), 256)
        config_logger.info(f"Max output length: {str(output_length)}")
        return output_length

    def prepare_dataset(self, extraction_data: ExtractionData):
        data_path = join(self.get_path(), "t5_transformers_data.csv")

        if exists(data_path):
            os.remove(data_path)

        text_inputs = [" ".join(sample.segment_selector_texts) for sample in extraction_data.samples]
        text_target = [sample.labeled_data.label_text if sample.labeled_data else "" for sample in extraction_data.samples]

        data = [
            [str(index + 1), f"Extract: {text_input}", text_target]
            for index, (text_input, text_target) in enumerate(zip(text_inputs, text_target))
        ]

        if not data:
            return None

        df = pd.DataFrame(data)
        df.columns = ["extraction_name", "input_with_prefix", "target"]
        df["not_used"] = ""

        os.makedirs(dirname(data_path), exist_ok=True)
        df.to_csv(data_path, quoting=csv.QUOTE_ALL)
        return data_path

    def train(self, extraction_data: ExtractionData):
        self.remove_method_data()
        train_path = self.prepare_dataset(extraction_data)

        if not train_path:
            return

        model_arguments = ModelArguments("HURIDOCS/mt5-small-spanish-es", cache_dir=self.get_cache_dir())
        output_length = self.get_max_output_length(extraction_data)
        self.get_max_length_path().write_text(str(output_length))
        data_training_arguments = DataTrainingArguments(
            train_file=train_path,
            validation_file=train_path,
            max_seq_length=self.get_max_input_length(extraction_data),
            max_answer_length=output_length,
            context_column="input_with_prefix",
            answer_column="target",
            question_column="not_used",
            version_2_with_negative=False,
        )

        t5_training_arguments = T5TrainingArguments(
            report_to=[],
            predict_with_generate=True,
            overwrite_output_dir=True,
            output_dir=self.get_model_path(),
            per_device_train_batch_size=1,
            learning_rate=5e-4,
            weight_decay=0.1,
            do_train=True,
            do_eval=True,
            do_predict=False,
            save_total_limit=2,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            early_stopping=True,
            num_train_epochs=30,
            early_stopping_patience=4,
            log_level="error",
            generation_max_length=output_length,
            use_cpu=not torch.cuda.is_available(),
        )

        run(model_arguments, data_training_arguments, t5_training_arguments)
        self.delete_checkpoints()

    @staticmethod
    def get_cache_dir():
        return join(DATA_PATH, "cache", "HF")

    def exists_model(self):
        return exists(self.get_model_path())

    def predict(self, predictions_samples: list[PredictionSample]) -> list[str]:
        texts = [" ".join(x.segment_selector_texts) for x in predictions_samples]

        if not self.exists_model():
            return texts

        samples = [TrainingSample(segment_selector_texts=sample.segment_selector_texts) for sample in predictions_samples]
        predict_data_path = self.prepare_dataset(ExtractionData(samples=samples))

        if not predict_data_path:
            return texts

        predictions = list()
        tokenizer = AutoTokenizer.from_pretrained("HURIDOCS/mt5-small-spanish-es")
        model = MT5ForConditionalGeneration.from_pretrained(self.get_model_path())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        max_length_predictions = int(self.get_max_length_path().read_text())
        config_logger.info(f"Max length predictions: {max_length_predictions}")
        for input_text, text in zip(pd.read_csv(predict_data_path)["input_with_prefix"].tolist(), texts):
            if not text.strip():
                predictions.append("")
                continue

            input_ids = tokenizer(input_text, return_tensors="pt").to(device).input_ids
            outputs = model.generate(input_ids, max_length=max_length_predictions)
            predictions.append(tokenizer.decode(outputs[0])[6:-4])

        return predictions

    @lru_cache(maxsize=1)
    def get_true_case(self):
        true_case_english_model_path = hf_hub_download(
            repo_id="HURIDOCS/spanish-truecasing",
            filename="english.dist",
            revision="69558da13436cc2b29d7db92d704976e2a7ffe16",
            cache_dir=join(DATA_PATH, "cache"),
        )

        true_case_spanish_model_path = hf_hub_download(
            repo_id="HURIDOCS/spanish-truecasing",
            filename="spanish.dist",
            revision="69558da13436cc2b29d7db92d704976e2a7ffe16",
            cache_dir=join(DATA_PATH, "cache"),
        )

        os.makedirs(join(DATA_PATH, "cache", "nltk_data"), exist_ok=True)
        return TrueCaser(true_case_english_model_path), TrueCaser(true_case_spanish_model_path)

    def get_true_case_segment_text(self, semantic_extraction_data):
        true_case_english, true_case_spanish = self.get_true_case()

        if semantic_extraction_data.language_iso == "en":
            return true_case_english.get_true_case(self.get_text_from_pdf_tags(semantic_extraction_data.texts))

        if semantic_extraction_data.language_iso == "es":
            return true_case_spanish.get_true_case(self.get_text_from_pdf_tags(semantic_extraction_data.texts))

        return self.get_text_from_pdf_tags(semantic_extraction_data.texts)

    def delete_checkpoints(self):
        for file_name in os.listdir(self.get_model_path()):
            if "checkpoint-" not in file_name:
                continue

            shutil.rmtree(join(self.get_model_path(), file_name), ignore_errors=True)
