from time import time

from sklearn.metrics import precision_score, recall_score
from py_markdown_table.markdown_table import markdown_table
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.use_cases.extractors.segment_selector.FastSegmentSelector import FastSegmentSelector
from trainable_entity_extractor.extractors.segment_selector.NearFastSegmentSelector import Near1FastSegmentSelector
from trainable_entity_extractor.use_cases.extractors.segment_selector.SegmentSelector import SegmentSelector
from trainable_entity_extractor.use_cases.extractors.segment_selector.SegmentSelectorBase import SegmentSelectorBase
from trainable_entity_extractor.use_cases.extractors.segment_selector.SegmentSelectorResults import SegmentSelectorResults
from trainable_entity_extractor.use_cases.extractors.segment_selector.get_data_for_performance import (
    get_data_for_performance,
)

extraction_identifier = ExtractionIdentifier(run_name="benchmark", extraction_name="segment_selector")

METHODS: list[SegmentSelectorBase] = [
    FastSegmentSelector(extraction_identifier),
    Near1FastSegmentSelector(extraction_identifier),
    SegmentSelector(extraction_identifier),
]


def get_train_test(pdfs_data):
    train_number = int(len(pdfs_data) * 0.5)
    train_data = pdfs_data[:train_number]
    test_data = pdfs_data[train_number:]
    return train_data, test_data


def print_results(results):
    results.sort(key=lambda x: (x.dataset, x.method))
    for method in METHODS:
        precisions = [x.precision for x in results if x.method == method.get_name()]
        recalls = [x.recall for x in results if x.method == method.get_name()]
        seconds = [x.seconds for x in results if x.method == method.get_name()]
        average_precision = round(sum(precisions) / len(precisions), 2)
        average_recall = round(sum(recalls) / len(recalls), 2)
        average_seconds = round(sum(seconds) / len(seconds))
        samples = sum([x.samples for x in results if x.method == method.get_name()])
        results.append(
            SegmentSelectorResults(
                method=method.get_name(),
                dataset="Average",
                samples=samples,
                precision=average_precision,
                recall=average_recall,
                seconds=average_seconds,
            )
        )

    data = [x.model_dump() for x in results]
    padding = SegmentSelectorResults.get_padding()
    markdown = markdown_table(data).set_params(padding_width=5, padding_weight=padding).get_markdown()
    print(markdown)


def get_performance_segment_selector():
    data = get_data_for_performance(filter_datasets=[])
    print(f"Datasets: {data.keys()}")

    results: list[SegmentSelectorResults] = list()
    for dataset, pdfs_data in data.items():
        training_set, test_set = get_train_test(pdfs_data)

        truth = [x.ml_label for pdf_data in test_set for x in pdf_data.pdf_data_segments]

        for segment in [x for pdf_data in test_set for x in pdf_data.pdf_data_segments]:
            segment.ml_label = 0

        for method in METHODS:
            method.prepare_model_folder()
            start = time()
            predicted_labels = method.get_predictions_for_performance(training_set, test_set)

            selector_results = SegmentSelectorResults(
                method=method.get_name(),
                dataset=dataset,
                samples=len(pdfs_data),
                precision=round(100 * precision_score(truth, predicted_labels), 2),
                recall=round(100 * recall_score(truth, predicted_labels), 2),
                seconds=round(time() - start),
            )

            results.append(selector_results)

    print_results(results)


if __name__ == "__main__":
    get_performance_segment_selector()
