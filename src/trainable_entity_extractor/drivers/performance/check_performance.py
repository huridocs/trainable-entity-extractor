from os.path import dirname, realpath, join
from typing import Type

import pandas as pd
from datetime import datetime

from trainable_entity_extractor.drivers.performance.Results import Results
from trainable_entity_extractor.extractors.text_to_text_extractor.TextToTextMethod import BaseMethod
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.MT5TrueCaseEnglishSpanishMethod import (
    MT5TrueCaseEnglishSpanishMethod,
)


class CheckPerformance:
    SCRIPT_PATH = dirname(realpath(__file__))

    TENANT = "check_performance"

    METHODS: list[Type[BaseMethod]] = [
        # T5ZeroShot,
        MT5TrueCaseEnglishSpanishMethod,
        # MT5EnglishSpanishMethod,
        # T5TransformersMethod,
        # T5Method,
        # SameInputOutputMethod,
        # RegexMethod,
        # DateParserMethod,
        # DateParserMethod,
        # FlanT5TrueCaseEnglishSpanishMethod
    ]

    DATASETS: list[str] = [
        # "code_spanish.tsv",
        # "country_spanish.tsv",
        "date_spanish.tsv",
        # "document_code.tsv",
        # "judge_name.tsv",
        # "vote_english.tsv",
        # "year_multilingual.tsv",
    ]

    def __init__(self, training_length):
        self.training_length = training_length
        self.semantic_information_data = None
        self.current_dataset = None
        self.current_method = None
        self.current_method_name = None
        self.current_prediction = None
        self.current_accuracy = None

        prefix = f"{datetime.now():%Y_%m_%d_%H_%M}"

        self.all_results = Results(results_name=f"all_results_{training_length}____{prefix}")
        self.best_results = Results(results_name=f"best_results_{training_length}____{prefix}")

    def dataset_name_to_extraction_id(self):
        return self.current_dataset.replace("_", "").replace(".tsv", "")

    def get_semantic_extraction_data(self, file_name):
        df = pd.read_csv(join(self.SCRIPT_PATH, "performance", "datasets", file_name), sep="\t")
        self.semantic_information_data = []
        for index, row in df.iterrows():
            semantic_extraction_data = SemanticExtractionData(
                text=row["output"], segment_text=row["input"], language_iso=row["language"]
            )
            self.semantic_information_data.append(semantic_extraction_data)

    def run(self):
        for self.current_dataset in self.DATASETS:
            self.get_semantic_extraction_data(self.current_dataset)
            self.best_results.set_start_time()

            accuracies = list()
            method_names = list()

            train, test = BaseMethod.get_train_test(self.semantic_information_data, self.training_length)

            for self.current_method in self.METHODS:
                self.run_method(accuracies)
                method_names.append(self.current_method_name)
                self.all_results.save_result(
                    dataset=self.current_dataset,
                    method=self.current_method_name,
                    accuracy=self.current_accuracy,
                    train_length=len(train),
                    test_length=len(test),
                )

            best_method = method_names[accuracies.index(max(accuracies))]
            self.best_results.save_result(
                dataset=self.current_dataset,
                method=best_method,
                accuracy=max(accuracies),
                train_length=len(train),
                test_length=len(test),
            )

        self.write_results()

    def run_method(self, accuracies):
        self.all_results.set_start_time()
        method_instance = self.current_method(self.TENANT, self.dataset_name_to_extraction_id())
        self.current_accuracy, self.current_prediction = method_instance.performance(
            self.semantic_information_data, self.training_length
        )
        self.write_mistakes()
        self.current_method_name = method_instance.get_name()
        accuracies.append(self.current_accuracy)

    def write_results(self):
        self.all_results.write_results()
        self.best_results.write_results()

    def write_mistakes(self):
        _, performance_test_set = BaseMethod.get_train_test(self.semantic_information_data, self.training_length)
        correct_path = "../performance_results/mistakes/"
        correct_path += f"{self.training_length}_{self.current_method_name}_{self.current_dataset}_correct.txt"
        mistakes_path = "../performance_results/mistakes/"
        mistakes_path += f"{self.training_length}_{self.current_method_name}_{self.current_dataset}_mistakes.txt"

        with open(mistakes_path, "w") as input_file:
            input_file.write("Mistakes\n\n")

        with open(correct_path, "w") as input_file:
            input_file.write("Correct\n\n")

        for prediction, semantic_information_data in zip(self.current_prediction, performance_test_set):
            text_one_line = semantic_information_data.segment_text.replace("\n", " ")

            file_name = correct_path if prediction == semantic_information_data.text else mistakes_path

            with open(file_name, "a") as input_file:
                line = f"prediction:{prediction}\ntarget:{semantic_information_data.text}\ntext:{text_one_line}\n\n"
                input_file.write(line)


if __name__ == "__main__":
    for i in [5]:
        check_performance = CheckPerformance(i)
        check_performance.run()
    #
    # check_performance = CheckPerformance(5)
    # check_performance.run()
