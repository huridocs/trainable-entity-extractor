import json
import os
import pickle
import random
from os import listdir
from os.path import join
from pathlib import Path
from time import sleep, time

import requests
from pdf_features.PdfFeatures import PdfFeatures
from sklearn.metrics import f1_score
from tqdm import tqdm

from trainable_entity_extractor.config import APP_PATH, ROOT_PATH
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.SegmentBox import SegmentBox
from trainable_entity_extractor.domain.SegmentationData import SegmentationData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.PdfMultiOptionMethod import (
    PdfMultiOptionMethod,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.PdfToMultiOptionExtractor import (
    PdfToMultiOptionExtractor,
)

PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH = join(
    Path(__file__).parent, "extractors", "pdf_to_multi_option_extractor", "labeled_data"
)
PDF_DATA_FOLDER_PATH = join(ROOT_PATH, "data", "pdf_data_cache")
LABELED_DATA_PATH = join(APP_PATH, "pdf_topic_classification", "labeled_data")

LABELED_DATA_PDFS_PATH = join(ROOT_PATH.parent, "pdf-labeled-data", "pdfs")

BASE_LINE = {
    "cejil_president": (100.0, "NextWordsTokenSelectorFuzzy75"),
    "cyrilla_keywords": (53.49, "FuzzyFirstCleanLabel"),
    "cejil_date": (20.83, "FuzzyAll88"),
    "cejil_countries": (69.05, "FuzzyFirstCleanLabel"),
    "d4la_document_type": (44.07, "CleanBeginningDotDigits500_SingleLabelSetFit"),
    "cejil_secretary": (80.0, "FuzzyAll75"),
    "countries_in_favor": (96.89, "PreviousWordsSentenceSelectorFuzzyCommas"),
    "countries_in_favor_empty_labels": (96.89, "PreviousWordsSentenceSelectorFuzzyCommas"),
    "cejil_judge": (92.86, "FuzzyLast"),
}


def get_task_pdf_names():
    task_pdf_names: dict[str, set[str]] = dict()

    for task_name in listdir(str(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH)):
        with open(join(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH, task_name, "labels.json"), mode="r") as file:
            labels_dict: dict[str, list[str]] = json.load(file)
            task_pdf_names.setdefault(task_name, set()).update(labels_dict.keys())

    return task_pdf_names


def cache_pdf_data(pdf_name: str, pickle_path: Path):
    pdf_features = PdfFeatures.from_poppler_etree(join(LABELED_DATA_PDFS_PATH, pdf_name, "etree.xml"))

    with open(join(LABELED_DATA_PDFS_PATH, pdf_name, "document.pdf"), "rb") as stream:
        files = {"file": stream}

        results = requests.post("http://localhost:5060", files=files)

    if results.status_code != 200:
        raise Exception("Error extracting the paragraphs")

    segments: list[SegmentBox] = [SegmentBox(**segment_box) for segment_box in results.json()]

    pdf_data = PdfData(pdf_features=pdf_features, file_name=pdf_name)
    segmentation_data = SegmentationData(
        page_width=segments[0].page_width,
        page_height=segments[0].page_height,
        xml_segments_boxes=segments,
        label_segments_boxes=[],
    )

    pdf_data.set_segments_from_segmentation_data(segmentation_data)

    os.makedirs(pickle_path.parent, exist_ok=True)
    with open(pickle_path, mode="wb") as file:
        pickle.dump(pdf_data, file)

    return pdf_data


def get_samples(task_name):
    with open(join(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH, task_name, "labels.json"), mode="r") as file:
        labels_dict: dict[str, list[str]] = json.load(file)

    multi_option_samples: list[TrainingSample] = list()
    for pdf_name in tqdm(sorted(get_task_pdf_names()[task_name])):
        pickle_path = join(PDF_DATA_FOLDER_PATH, f"{pdf_name}.pickle")

        if Path(pickle_path).exists():
            with open(pickle_path, mode="rb") as file:
                pdf_data: PdfData = pickle.load(file)
        else:
            pdf_data: PdfData = cache_pdf_data(pdf_name, Path(pickle_path))

        values = [Option(id=x, label=x) for x in labels_dict[pdf_name]]
        language_iso = "es" if "cejil" in task_name else "en"

        extraction_sample = TrainingSample(
            pdf_data=pdf_data, labeled_data=LabeledData(values=values, language_iso=language_iso)
        )
        multi_option_samples.append(extraction_sample)

    random.seed(42)
    random.shuffle(multi_option_samples)
    return multi_option_samples


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


def performance_report():
    f1s_method_name = get_f1_scores_method_names()
    sleep(1)
    print()
    print("REPORT:")
    print("-------")
    for key, (value, method_name) in f1s_method_name.items():
        if value < BASE_LINE[key][0]:
            print(f"{key}: PERFORMANCE DECREASED!!!!!")
        else:
            print(f"{key}: Good performance")

        print(f"Base performance: {BASE_LINE[key][0]}% with method {BASE_LINE[key][1]}")
        print(f"Performance: {value}% with method {method_name}")
        print()


def get_f1_scores_method_names() -> dict[str, (float, str)]:
    f1s_method_name = dict()
    for dataset in get_multi_option_benchmark_data(filter_by=[]):
        truth_one_hot, prediction_one_hot, method_name, _ = get_predictions(dataset)
        f1 = round(100 * f1_score(truth_one_hot, prediction_one_hot, average="micro"), 2)
        f1s_method_name[dataset.extraction_identifier.extraction_name] = (f1, method_name)

    return f1s_method_name


def get_predictions(dataset: ExtractionData) -> (list[list[int]], list[list[int]], str):
    training_samples_number = int(len(dataset.samples) * 0.5) if len(dataset.samples) > 10 else 10
    training_samples = dataset.samples[:training_samples_number]
    test_samples = dataset.samples[training_samples_number:] if len(dataset.samples) > 20 else dataset.samples

    training_dataset = ExtractionData(
        samples=training_samples,
        options=dataset.options,
        multi_value=dataset.multi_value,
        extraction_identifier=dataset.extraction_identifier,
    )
    extractor = PdfToMultiOptionExtractor(dataset.extraction_identifier)
    extractor.create_model(training_dataset)
    prediction_samples = [PredictionSample(pdf_data=sample.pdf_data) for sample in test_samples]
    context_samples, predictions = extractor.get_predictions(prediction_samples)
    values_list = [x.labeled_data.values for x in test_samples]
    truth_one_hot = PdfMultiOptionMethod.one_hot_to_options_list(values_list, dataset.options)
    prediction_one_hot = PdfMultiOptionMethod.one_hot_to_options_list(predictions, dataset.options)
    return truth_one_hot, prediction_one_hot, extractor.get_best_method(training_dataset).get_name(), context_samples


def get_mistakes() -> dict[str, (float, str)]:
    f1s_method_name = dict()
    for dataset in get_multi_option_benchmark_data(filter_by=[]):
        truth_one_hot, prediction_one_hot, method_name, test_samples = get_predictions(dataset)

        correct = 0
        mistakes = 0
        for truth, prediction, sample in zip(truth_one_hot, prediction_one_hot, test_samples):
            text = " ".join([x.text_cleaned for x in sample.pdf_data.pdf_data_segments if x.ml_label])
            missing = [dataset.options[i].label for i in range(len(truth)) if truth[i] and not prediction[i]]
            wrong = [dataset.options[i].label for i in range(len(truth)) if not truth[i] and prediction[i]]

            if missing or wrong:
                print()
                print(f"PDF: {sample.pdf_data.file_name}")
                print(f"Text: {text}")
                print(f"Missing: {missing}")
                print(f"Wrong: {wrong}")
                mistakes += 1
            else:
                correct += 1

        print(f"\n\nCorrect predictions for: {correct} PDFs")
        print(f"Incorrect predictions for {mistakes} PDFs")

    return f1s_method_name


if __name__ == "__main__":
    start = time()
    print("start")
    performance_report()
    print("time", round(time() - start, 2), "s")
