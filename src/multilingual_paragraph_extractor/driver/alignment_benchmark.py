import json
from pathlib import Path
from py_markdown_table.markdown_table import markdown_table

from multilingual_paragraph_extractor.driver.AlignmentResult import AlignmentResult
from multilingual_paragraph_extractor.driver.Labels import Labels
from multilingual_paragraph_extractor.driver.label_data import get_algorithm_labels, LABELED_DATA_PATH


def get_f1_score(truth_paragraphs, prediction_paragraphs):
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for truth_paragraph in truth_paragraphs:
        if truth_paragraph in prediction_paragraphs:
            true_positive += 1
        else:
            false_negative += 1

    for prediction_paragraph in prediction_paragraphs:
        if prediction_paragraph not in truth_paragraphs:
            false_positive += 1

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return round(100 * precision, 2), round(100 * recall, 2), round(100 * f1_score, 2)


def get_average(alignment_results: list[AlignmentResult]) -> AlignmentResult:
    total_paragraphs = sum([x.total_paragraphs for x in alignment_results])
    total_seconds = sum([x.seconds for x in alignment_results])
    total_precision = sum([x.precision for x in alignment_results])
    total_recall = sum([x.recall for x in alignment_results])
    total_f1_score = sum([x.f1_score for x in alignment_results])

    return AlignmentResult(
        name="Average",
        algorithm=alignment_results[0].algorithm,
        precision=round(total_precision / len(alignment_results), 2),
        recall=round(total_recall / len(alignment_results), 2),
        f1_score=round(total_f1_score / len(alignment_results), 2),
        total_paragraphs=total_paragraphs,
        seconds=total_seconds,
    )


def get_alignment_benchmark():
    predictions_labels, times = get_algorithm_labels()

    results: list[AlignmentResult] = list()
    for prediction_labels, result_time in zip(predictions_labels, times):
        json_labels = json.loads(Path(LABELED_DATA_PATH, "labels", prediction_labels.get_label_file_name()).read_text())
        truth_labels = Labels(**json_labels)
        precision, recall, f1_score = get_f1_score(truth_labels.paragraphs, prediction_labels.paragraphs)
        results.append(
            AlignmentResult(
                name=prediction_labels.get_label_file_name().rsplit(".", 1)[0],
                algorithm="base_6_Feb_2025",
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                total_paragraphs=len(truth_labels.paragraphs),
                seconds=result_time,
            )
        )
    results.append(get_average(results))
    markdown = markdown_table([x.model_dump() for x in results]).set_params(padding_width=5).get_markdown()
    print(markdown)


if __name__ == "__main__":
    get_alignment_benchmark()
