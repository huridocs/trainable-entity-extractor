import importlib
import json
import os
import pickle
import random
import shutil

from datetime import datetime
from os.path import join, exists
from pathlib import Path

import requests
from pdf_features.PdfFeatures import PdfFeatures
from pdf_token_type_labels.PdfLabels import PdfLabels
from pdf_token_type_labels.TaskMistakes import TaskMistakes
from sklearn.metrics import f1_score

from trainable_entity_extractor.config import ROOT_PATH, DATA_PATH
from trainable_entity_extractor.domain.SegmentBox import SegmentBox
from trainable_entity_extractor.domain.SegmentationData import SegmentationData
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.drivers.performance.Results import Results
from trainable_entity_extractor.adapters.extractors.segment_selector.Paragraphs import Paragraphs
from trainable_entity_extractor.adapters.extractors.segment_selector.evaluate_config import (
    SIZES,
    SEED,
    LABELED_DATA_TO_USE,
    METHODS_TO_EXECUTE,
    PDF_LABELED_DATA_PATH,
)

RANDOM_SEED = 42


def get_segmentation_data(pdf_path: str, pdf_name: str) -> SegmentationData:
    pickle_path = join(DATA_PATH, "cached_segments", pdf_name + ".pickle")

    if exists(pickle_path):
        with open(pickle_path, "rb") as f:
            paragraphs = pickle.load(f)
    else:
        files = {
            "file": open(
                pdf_path,
                "rb",
            ),
        }
        response = requests.post("http://localhost:5051", files=files)
        paragraphs = Paragraphs(**response.json())
        with open(pickle_path, "wb") as f:  # open a text file
            pickle.dump(paragraphs, f)

    xml_segments_boxes = [
        SegmentBox(
            left=paragraph.left,
            top=paragraph.top,
            width=paragraph.width,
            height=paragraph.height,
            page_width=paragraphs.page_width,
            page_height=paragraphs.page_height,
            page_number=paragraph.page_number,
            segment_type=paragraph.type,
        )
        for paragraph in paragraphs.paragraphs
    ]

    return SegmentationData(
        page_width=paragraphs.page_width,
        page_height=paragraphs.page_height,
        xml_segments_boxes=xml_segments_boxes,
        label_segments_boxes=[],
    )


def load_pdf_segments(task: str, pdf_name: str) -> PdfData:
    labeled_data_root_path = join(ROOT_PATH.parent, "pdf-labeled-data")

    pdfs_path = join(labeled_data_root_path, "pdfs")
    pdf_features = PdfFeatures.from_poppler_etree(join(pdfs_path, pdf_name, "etree.xml"), pdf_name)

    pdf_path = join(pdfs_path, pdf_name, "document.pdf")
    segmentation_data: SegmentationData = get_segmentation_data(pdf_path, pdf_name)

    labeled_data_path = join(labeled_data_root_path, "labeled_data", "paragraph_selector", task, pdf_name, "labels.json")
    pdf_labels = PdfLabels(**json.loads(Path(labeled_data_path).read_text()))
    segmentation_data.label_segments_boxes = [
        SegmentBox(
            left=label.left,
            top=label.top,
            width=label.width,
            height=label.height,
            page_width=pdf_features.pages[0].page_width,
            page_height=pdf_features.pages[0].page_height,
            page_number=page.number,
        )
        for page in pdf_labels.pages
        for label in page.labels
    ]

    pdf_segments = PdfData(pdf_features=pdf_features)
    pdf_segments.set_segments_from_segmentation_data(segmentation_data)
    pdf_segments.set_ml_label_from_segmentation_data(segmentation_data)

    return pdf_segments


def load_training_testing_data(task: str, seed: int) -> (list[PdfData], list[PdfData]):
    print()
    print("Loading data for", task, "with seed", seed)

    labeled_data_path = join(ROOT_PATH.parent, "pdf-labeled-data", "labeled_data", "paragraph_selector", task)
    pdfs_segments = list()
    for index, pdf_name in enumerate(os.listdir(labeled_data_path)):
        pdfs_segments.append(load_pdf_segments(task, pdf_name))

    current_pdfs_segments = [x for x in pdfs_segments]
    random.seed(seed)
    random.shuffle(current_pdfs_segments)

    train_size = int(len(current_pdfs_segments) * 0.6)

    return current_pdfs_segments[:train_size], current_pdfs_segments[train_size:]


def get_loop_values():
    for size in SIZES:
        for task in LABELED_DATA_TO_USE:
            yield size, SEED, task


def snake_case_to_pascal_case(name: str):
    return "".join(word.title() for word in name.split("_"))


def save_mistakes(method_name: str, task: str, testing_pdfs_segments: list[PdfData], predictions_binary: list[int]):
    prediction_index = 0
    for pdf_segments in testing_pdfs_segments:
        y_true = [segment.ml_label for segment in pdf_segments.pdf_data_segments]
        pdf_segments_predictions = predictions_binary[prediction_index : prediction_index + len(y_true)]
        prediction_index += len(y_true)

        task_mistakes = TaskMistakes(PDF_LABELED_DATA_PATH, task + "_" + method_name, pdf_segments.pdf_features.file_name)
        for segment, truth, prediction in zip(pdf_segments.pdf_data_segments, y_true, pdf_segments_predictions):
            if not truth and not prediction:
                continue
            task_mistakes.add(segment.page_number, segment.bounding_box, truth, prediction)

        task_mistakes.save()


def run_one_method(
    method_name: str,
    task: str,
    training_pdfs_segments: list[PdfData],
    testing_pdfs_segments: list[PdfData],
    results: Results,
):
    results.set_start_time()
    method_class_name = snake_case_to_pascal_case(method_name)
    import_from = f"segment_selector.methods.{method_name}.{method_class_name}"

    model_path = Path(join(ROOT_PATH, "models_data", "segment_selector", task, "model"))

    shutil.rmtree(model_path.parent, ignore_errors=True)
    os.makedirs(model_path.parent)

    method = importlib.import_module(import_from, method_class_name)
    method_class = getattr(method, method_class_name)
    method_instance = method_class()
    model = method_instance.can_be_used(training_pdfs_segments, model_path)
    predictions = method_instance.predict(model, testing_pdfs_segments, model_path)

    y_true = [x.ml_label for test in testing_pdfs_segments for x in test.pdf_data_segments]
    prediction_binary = [1 if prediction > 0.5 else 0 for prediction in predictions]

    save_mistakes(method_name, task, testing_pdfs_segments, prediction_binary)

    f1 = round(100 * f1_score(y_true, prediction_binary), 2)

    results.save_result(
        dataset=task,
        method=method_name,
        accuracy=f1,
        train_length=len(training_pdfs_segments),
        test_length=len(testing_pdfs_segments),
    )

    return f1


def evaluate_methods():
    results_name = f"paragraph_selector_{datetime.now():%Y_%m_%d_%H_%M}"
    results = Results(results_name)

    for method_name in METHODS_TO_EXECUTE:
        f1s = list()
        for size, seed, task in get_loop_values():
            training_pdfs_segments, testing_pdfs_segments = load_training_testing_data(task, seed)
            training_pdfs_segments = training_pdfs_segments[:size]

            print(
                f"\n\nevaluating time:{datetime.now():%Y/%m/%d %H:%M} size:{size} seed:{seed} task:{task} method:{method_name}"
            )
            f1 = run_one_method(method_name, task, training_pdfs_segments, testing_pdfs_segments, results)
            f1s.append(f1)

        results.set_start_time()
        results.save_result(
            dataset="Average",
            method="",
            accuracy=round((sum(f1s) + 83) / (len(f1s) + 1), 2),
            train_length=0,
            test_length=0,
        )

    results.write_results()


if __name__ == "__main__":
    evaluate_methods()
