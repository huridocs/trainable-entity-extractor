import json
from pathlib import Path

from pdf_annotate import PdfAnnotator, Location, Appearance
from py_markdown_table.markdown_table import markdown_table
from visualization.save_output_to_pdf import hex_color_to_rgb

from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures
from multilingual_paragraph_extractor.driver.AlignmentResult import AlignmentResult
from multilingual_paragraph_extractor.driver.Labels import Labels
from multilingual_paragraph_extractor.driver.label_data import (
    get_algorithm_labels,
    LABELED_DATA_PATH,
    PARAGRAPH_EXTRACTION_PATH,
)


def add_annotation(annotator: PdfAnnotator, paragraph_features: ParagraphFeatures, text: str, color: str):
    left, top, right, bottom = (
        paragraph_features.bounding_box.left,
        paragraph_features.page_height - paragraph_features.bounding_box.top,
        paragraph_features.bounding_box.right,
        paragraph_features.page_height - paragraph_features.bounding_box.bottom,
    )

    text_box_size = 20 * 8 + 8

    annotator.add_annotation(
        "square",
        Location(x1=left, y1=bottom, x2=right, y2=top, page=paragraph_features.page_number - 1),
        Appearance(stroke_color=hex_color_to_rgb(color)),
    )

    annotator.add_annotation(
        "square",
        Location(x1=left, y1=top, x2=left + text_box_size, y2=top + 10, page=paragraph_features.page_number - 1),
        Appearance(fill=hex_color_to_rgb(color)),
    )

    annotator.add_annotation(
        "text",
        Location(x1=left, y1=top, x2=left + text_box_size, y2=top + 10, page=paragraph_features.page_number - 1),
        Appearance(content=text, font_size=8, fill=(1, 1, 1), stroke_width=3),
    )


def show_mistakes(truth_labels: Labels, prediction_labels: Labels):
    pdf_path = Path(LABELED_DATA_PATH, "pdfs", truth_labels.get_main_pdf_name())
    output_pdf_path = Path(PARAGRAPH_EXTRACTION_PATH, "mistakes", truth_labels.get_mistakes_pdf_name())

    annotator = PdfAnnotator(str(pdf_path))

    for prediction_paragraph, alignment_score in zip(prediction_labels.paragraphs, prediction_labels.get_alignment_scores()):
        text = str(int(100 * alignment_score.score)) + "% "
        text += " ".join([x for x in alignment_score.other_paragraph.text_cleaned.split()][:3])
        color = "#008B8B" if prediction_paragraph in truth_labels.paragraphs else "#F15628"
        add_annotation(annotator, alignment_score.main_paragraph, text, color)

    if not output_pdf_path.parent.exists():
        output_pdf_path.parent.mkdir(parents=True)

    annotator.write(output_pdf_path)


def get_f1_score(truth_labels: Labels, prediction_labels: Labels):
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for truth_paragraph in truth_labels.paragraphs:
        if truth_paragraph in prediction_labels.paragraphs:
            true_positive += 1
        else:
            false_negative += 1

    for prediction_paragraph in prediction_labels.paragraphs:
        if prediction_paragraph not in truth_labels.paragraphs:
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
    predictions_labels = get_algorithm_labels()

    results: list[AlignmentResult] = list()
    for prediction_labels in predictions_labels:
        json_labels = json.loads(Path(LABELED_DATA_PATH, "labels", prediction_labels.get_label_file_name()).read_text())
        truth_labels = Labels(**json_labels)
        precision, recall, f1_score = get_f1_score(truth_labels, prediction_labels)
        show_mistakes(truth_labels, prediction_labels)
        results.append(
            AlignmentResult(
                name=prediction_labels.get_label_file_name().rsplit(".", 1)[0],
                algorithm="lightGBM_6_Feb_2025",
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                total_paragraphs=len(truth_labels.paragraphs),
                seconds=prediction_labels.get_seconds(),
            )
        )
    results.append(get_average(results))
    markdown = markdown_table([x.model_dump() for x in results]).set_params(padding_width=5).get_markdown()
    print(markdown)


if __name__ == "__main__":
    get_alignment_benchmark()
