import json
import pickle
from os import listdir
from os.path import join
from pathlib import Path
from time import time

import rich
from sklearn.metrics import f1_score

from trainable_entity_extractor.config import ROOT_PATH, APP_PATH
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.LabeledData import LabeledData
from trainable_entity_extractor.data.Option import Option
from trainable_entity_extractor.data.PdfData import PdfData
from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.extractors.ExtractorBase import ExtractorBase
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.PdfMultiOptionMethod import PdfMultiOptionMethod

from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.PdfToMultiOptionExtractor import (
    PdfToMultiOptionExtractor,
)
from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.TrainingSample import TrainingSample
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyFirstCleanLabel import (
    FuzzyFirstCleanLabel,
)

from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.results import (
    get_results_table,
    add_row,
    add_prediction_row,
    get_predictions_table,
)

PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH = join(
    Path(__file__).parent, "extractors", "pdf_to_multi_option_extractor", "labeled_data"
)
PDF_DATA_FOLDER_PATH = join(ROOT_PATH, "data", "pdf_data_cache")
LABELED_DATA_PATH = join(APP_PATH, "pdf_topic_classification", "labeled_data")

# text_extractors = [CleanBeginningDigits3000, CleanEndDot1000]
# multi_option_extractors = [TfIdfMethod]
# PDF_TOPIC_CLASSIFICATION_METHODS = [PdfMultiOptionMethod(x, y) for x in text_extractors for y in multi_option_extractors]


PDF_TOPIC_CLASSIFICATION_METHODS = [
    # FuzzyFirst(),
    # FuzzyLast(),
    FuzzyFirstCleanLabel(),
    # FuzzyLastCleanLabel(),
    # FuzzyAll75(),
    # FuzzyAll88(),
    # FuzzyAll100(),
]


def get_multi_option_benchmark_data(filter_by: list[str] = None) -> list[ExtractionData]:
    benchmark_data: list[ExtractionData] = list()
    for task_name in listdir(str(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH)):
        if filter_by and task_name not in filter_by:
            continue

        print(f"Loading task {task_name}")

        with open(join(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH, task_name, "options.json"), mode="r") as file:
            options = [Option(id=x, label=x) for x in json.load(file)]

        multi_option_samples = get_samples(task_name)
        multi_value: bool = len([sample for sample in multi_option_samples if len(sample.labeled_data.values) > 1]) != 0
        extraction_identifier = ExtractionIdentifier(run_name="benchmark", extraction_name=task_name)
        benchmark_data.append(
            ExtractionData(
                samples=multi_option_samples,
                options=options,
                multi_value=multi_value,
                extraction_identifier=extraction_identifier,
            )
        )

    return benchmark_data


def get_samples(task_name):
    with open(join(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH, task_name, "labels.json"), mode="r") as file:
        labels_dict: dict[str, list[str]] = json.load(file)

    multi_option_samples: list[TrainingSample] = list()
    for pdf_name in sorted(get_task_pdf_names()[task_name]):
        with open(join(PDF_DATA_FOLDER_PATH, f"{pdf_name}.pickle"), mode="rb") as file:
            pdf_data: PdfData = pickle.load(file)

        values = [Option(id=x, label=x) for x in labels_dict[pdf_name]]
        language_iso = "es" if "cejil" in task_name else "en"

        extraction_sample = TrainingSample(
            pdf_data=pdf_data, labeled_data=LabeledData(values=values, language_iso=language_iso)
        )
        multi_option_samples.append(extraction_sample)
    return multi_option_samples


def get_task_pdf_names():
    task_pdf_names: dict[str, set[str]] = dict()

    for task_name in listdir(str(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH)):
        with open(join(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH, task_name, "labels.json"), mode="r") as file:
            labels_dict: dict[str, list[str]] = json.load(file)
            task_pdf_names.setdefault(task_name, set()).update(labels_dict.keys())

    return task_pdf_names


def loop_datasets_methods(datasets: list[str] = None):
    extractions_data: list[ExtractionData] = get_multi_option_benchmark_data(datasets)

    for extraction_data in extractions_data:
        for method in PDF_TOPIC_CLASSIFICATION_METHODS:
            yield extraction_data, method


def get_benchmark_custom_methods(repetitions: int = 4):
    results_table = get_results_table()

    # cejil_countries
    # cejil_date
    # cejil_judge
    # cejil_president
    # cejil_secretary
    # cyrilla_keywords
    # d4la_document_type
    for extraction_data, method in loop_datasets_methods(["cejil_countries"]):
        start = time()
        print("Calculating", extraction_data.extraction_identifier, method.get_name())
        performance = method.get_performance(extraction_data, extraction_data)
        add_row(results_table, method, round(time() - start), performance)


def get_multi_option_extractor_benchmark():
    results_table = get_results_table()

    # cejil_countries
    # cejil_date
    # cejil_judge
    # cejil_president
    # cejil_secretary
    # cyrilla_keywords
    # d4la_document_type
    extractions_data: list[ExtractionData] = get_multi_option_benchmark_data(["cyrilla_keywords"])
    for extraction_data in extractions_data:
        start = time()
        extractor = PdfToMultiOptionExtractor(extraction_identifier=extraction_data.extraction_identifier)
        train_set, test_set = ExtractorBase.get_train_test_sets(extraction_data)
        values_list = [x.labeled_data.values for x in test_set.samples]
        truth_one_hot = PdfMultiOptionMethod.one_hot_to_options_list(values_list, extraction_data.options)
        extractor.create_model(train_set)

        test_data = [PredictionSample(pdf_data=x.pdf_data) for x in test_set.samples]
        suggestions = extractor.get_suggestions(test_data)
        values_list = [x.values for x in suggestions]
        predictions_one_hot = PdfMultiOptionMethod.one_hot_to_options_list(values_list, extraction_data.options)

        performance = 100 * f1_score(truth_one_hot, predictions_one_hot, average="micro")

        results_table.add_row(
            "Extractor",
            extraction_data.extraction_identifier.extraction_name,
            f"{round((time() - start) / 60, 1)}",
            f"{round(performance, 2)}%",
        )
        rich.print(results_table)


def check_results():
    prediction_table = get_predictions_table()
    extractions_data: list[ExtractionData] = get_multi_option_benchmark_data(["cejil_countries"])
    for extraction_data in extractions_data:
        extractor = PdfToMultiOptionExtractor(extraction_identifier=extraction_data.extraction_identifier)

        print(f"Calculating {extractor.extraction_identifier} {extractor.get_name()}")
        train_set, test_set = ExtractorBase.get_train_test_sets(extraction_data)
        labels = [x.labeled_data.values for x in test_set.samples]
        test_data = [PredictionSample(pdf_data=x.pdf_data) for x in test_set.samples]
        suggestions = extractor.get_suggestions(test_data)
        pdfs_names = [x.pdf_data.file_name for x in test_set.samples]
        predictions = [x.values for x in suggestions]
        for label, prediction, pdf_name in zip(labels, predictions, pdfs_names):
            add_prediction_row(prediction_table, pdf_name, label, prediction)

        add_prediction_row(prediction_table)

    rich.print(prediction_table)


if __name__ == "__main__":
    # get_benchmark_custom_methods(4)
    get_multi_option_extractor_benchmark()
    # check_results()
