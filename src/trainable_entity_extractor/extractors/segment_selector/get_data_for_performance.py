import json
import pickle
from os import listdir
from os.path import join
from pathlib import Path

from trainable_entity_extractor.config import ROOT_PATH
from trainable_entity_extractor.data.PdfData import PdfData
from trainable_entity_extractor.data.SegmentBox import SegmentBox
from trainable_entity_extractor.data.SegmentationData import SegmentationData
from performance_pdf_to_multi_option_report import PDF_DATA_FOLDER_PATH, cache_pdf_data

LABELED_DATA_PATH = Path(ROOT_PATH.parent, "pdf-labeled-data", "labeled_data", "paragraph_selector")

DATASETS = [
    "date",
    "decides",
    "first_paragraph_having_seen",
    "plan_many_date",
    "plan_many_title",
    "president",
    "president_empty_labels",
    "rightdocs_titles",
    "secretary",
    "semantic_president",
    "signatories",
]


def get_data_for_performance(filter_datasets: list[str] = None) -> dict[str, list[PdfData]]:
    if filter_datasets:
        filtered_datasets = [x for x in DATASETS if x in filter_datasets]
    else:
        filtered_datasets = DATASETS

    pdf_data_per_dataset = {}
    for dataset in filtered_datasets:
        pdf_data_per_dataset[dataset] = []
        for pdf_name in listdir(Path(LABELED_DATA_PATH, dataset)):
            pickle_path = join(PDF_DATA_FOLDER_PATH, f"{pdf_name}.pickle")
            if Path(pickle_path).exists():
                with open(pickle_path, mode="rb") as file:
                    pdf_data: PdfData = pickle.load(file)
            else:
                pdf_data: PdfData = cache_pdf_data(pdf_name, Path(pickle_path))

            segmentation_data = SegmentationData(
                page_width=0, page_height=0, xml_segments_boxes=[], label_segments_boxes=get_labels(dataset, pdf_name)
            )
            pdf_data.set_ml_label_from_segmentation_data(segmentation_data)
            pdf_data_per_dataset[dataset].append(pdf_data)

    return pdf_data_per_dataset


def get_labels(dataset, pdf_name):
    label_segments_boxes = []
    labels = json.loads(Path(LABELED_DATA_PATH, dataset, pdf_name, "labels.json").read_text())
    for page in labels["pages"]:
        for label in page["labels"]:
            label_segments_boxes.append(
                SegmentBox(
                    left=label["left"],
                    top=label["top"],
                    width=label["width"],
                    height=label["height"],
                    page_number=page["number"],
                )
            )
    return label_segments_boxes
