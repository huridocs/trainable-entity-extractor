import json
from os import listdir
from os.path import join
from pathlib import Path
from time import time

import rich
from sklearn.metrics import f1_score

from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.LabeledData import LabeledData
from trainable_entity_extractor.data.Option import Option
from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.data.TrainingSample import TrainingSample
from trainable_entity_extractor.extractors.ExtractorBase import ExtractorBase
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.PdfMultiOptionMethod import PdfMultiOptionMethod
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.results import get_results_table
from trainable_entity_extractor.extractors.text_to_multi_option_extractor.TextToMultiOptionExtractor import (
    TextToMultiOptionExtractor,
)

LABELED_DATA_PATH = join(Path(__file__).parent, "extractors", "text_to_multi_option_extractor", "labeled_data")


def get_extraction_data(filter_by: list[str] = None):
    extractions_data = list()
    for task_name in listdir(str(LABELED_DATA_PATH)):
        if filter_by and task_name not in filter_by:
            continue

        extraction_identifier = ExtractionIdentifier(run_name="text_benchmark", extraction_name=task_name)
        texts_values = json.loads(Path(LABELED_DATA_PATH, task_name, "labels.json").read_text())
        options_text = Path(LABELED_DATA_PATH, task_name, "options.json").read_text()
        options = [Option(id=x, label=x) for x in json.loads(options_text)]

        samples: list[TrainingSample] = list()
        for i, text_value in enumerate(texts_values):
            values = [Option(id=x, label=x) for x in text_value["values"]]
            language_iso = "es" if "cejil" in task_name else "en"
            labeled_data = LabeledData(values=values, entity_name=str(i), language_iso=language_iso)
            extraction_sample = TrainingSample(tags_texts=[text_value["text"]], labeled_data=labeled_data)
            samples.append(extraction_sample)

        multi_value: bool = len([sample for sample in samples if len(sample.labeled_data.values) > 1]) > 0

        extractions_data.append(
            ExtractionData(
                options=options, samples=samples, multi_value=multi_value, extraction_identifier=extraction_identifier
            )
        )

    return extractions_data


def get_benchmark():
    results_table = get_results_table()

    # cejil_countries
    # cejil_judge
    # cejil_president
    # cejil_secretary
    # cyrilla_keywords
    # d4la_document_type
    # cejil_mechanism
    # cejil_descriptors
    # rightdocs_vote_type
    # action
    # themes
    # issues
    extractions_data: list[ExtractionData] = get_extraction_data(filter_by=["issues"])
    for extraction_data in extractions_data:
        start = time()
        extractor = TextToMultiOptionExtractor(extraction_identifier=extraction_data.extraction_identifier)
        train_set, test_set = ExtractorBase.get_train_test_sets(extraction_data, limit_samples=False)
        values_list = [x.labeled_data.values for x in test_set.samples]
        truth_one_hot = PdfMultiOptionMethod.one_hot_to_options_list(values_list, extraction_data.options)
        extractor.create_model(train_set)

        tags_texts = [x.tags_texts for x in test_set.samples]
        test_data = [PredictionSample(tags_texts=tag_text, entity_name=str(i)) for i, tag_text in enumerate(tags_texts)]
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
    # cejil_countries
    # cejil_judge
    # cejil_president
    # cejil_secretary
    # cyrilla_keywords
    # d4la_document_type
    # cejil_mechanism
    # cejil_descriptors
    # rightdocs_vote_type
    extractions_data: list[ExtractionData] = get_extraction_data(filter_by=[""])
    for extraction_data in extractions_data:
        extractor = TextToMultiOptionExtractor(extraction_identifier=extraction_data.extraction_identifier)
        train_set, test_set = ExtractorBase.get_train_test_sets(extraction_data, limit_samples=False)
        test_data = [
            PredictionSample(tags_texts=x.tags_texts, entity_name=x.labeled_data.entity_name) for x in test_set.samples
        ]
        suggestions = extractor.get_suggestions(test_data)
        for suggestion, sample in zip(suggestions, test_set.samples):
            print()
            print(" ".join([x for x in sample.tags_texts]).replace("\n", " "))
            print([x.label for x in sample.labeled_data.values])
            print([x.label for x in suggestion.values])


if __name__ == "__main__":
    get_benchmark()
    # check_results()
