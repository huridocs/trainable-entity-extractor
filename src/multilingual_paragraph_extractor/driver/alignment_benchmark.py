import json
from pathlib import Path

from pdf_annotate import PdfAnnotator, Location, Appearance
from py_markdown_table.markdown_table import markdown_table
from rapidfuzz import fuzz
from visualization.save_output_to_pdf import hex_color_to_rgb

from multilingual_paragraph_extractor.domain.AlignmentScore import AlignmentScore
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


def print_with_breaks(text: str, words_per_line: int = 20) -> str:
    words = text.split()
    lines = [" ".join(words[i : i + words_per_line]) for i in range(0, len(words), words_per_line)]
    return "\n".join(lines)


def fix_labels(truth_labels: Labels, alignment_score: AlignmentScore):
    for truth_paragraph in truth_labels.paragraphs:
        fuzz_ratio_main = round(fuzz.ratio(alignment_score.main_paragraph.original_text, truth_paragraph.main_language))
        fuzz_ratio_other = round(fuzz.ratio(alignment_score.other_paragraph.original_text, truth_paragraph.other_language))
        if fuzz_ratio_main > 95 or fuzz_ratio_other > 95:
            base_file_name = truth_labels.main_xml_name.rsplit("_", 1)[0]
            main_json_file_name = base_file_name + "_" + truth_labels.main_language
            main_json_file_name += "_" + truth_labels.other_language + ".json"
            other_json_file_name = base_file_name + "_" + truth_labels.other_language
            other_json_file_name += "_" + truth_labels.main_language + ".json"
            print(f"\nFOR MAIN DOCUMENT: ({main_json_file_name})")
            print(f'\033[95m"main_language": "{alignment_score.main_paragraph.original_text}",\033[0m')
            print(f'\033[92m"other_language": "{alignment_score.other_paragraph.original_text}"\033[0m\n')

            print(f"\nFOR OTHER DOCUMENT: ({other_json_file_name})")
            print(f'\033[94m"main_language": "{alignment_score.other_paragraph.original_text}",\033[0m')
            print(f'\033[91m"other_language": "{alignment_score.main_paragraph.original_text}"\033[0m\n')

            print("\033[95m" + print_with_breaks(alignment_score.main_paragraph.original_text) + "\033[0m")
            print("\033[92m" + print_with_breaks(alignment_score.other_paragraph.original_text) + "\033[0m\n")

    print("*" * 30)


def save_mistakes(truth_labels: Labels, prediction_labels: Labels):
    pdf_path = Path(LABELED_DATA_PATH, "pdfs", truth_labels.get_main_pdf_name())
    output_pdf_path = Path(PARAGRAPH_EXTRACTION_PATH, "mistakes", truth_labels.get_mistakes_pdf_name())

    annotator = PdfAnnotator(str(pdf_path))

    for prediction_paragraph, alignment_score in zip(prediction_labels.paragraphs, prediction_labels.get_alignment_scores()):
        text = str(int(100 * alignment_score.score)) + "% "
        text += " ".join([x for x in alignment_score.other_paragraph.text_cleaned.split()][:3])

        if prediction_paragraph in truth_labels.paragraphs:
            color = "#008B8B"
        else:
            print()
            print(f"Prediction not in truth")
            print(alignment_score.main_paragraph.original_text)
            print(alignment_score.other_paragraph.original_text)
            color = "#F15628"
            fix_labels(truth_labels, alignment_score)

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
    seconds = sum([x.seconds for x in alignment_results]) / len(alignment_results)
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
        seconds=round(seconds, 2),
    )


def get_alignment_benchmark(model_name: str, show_mistakes: bool = True):
    predictions_labels = get_algorithm_labels(["cejil_1"])

    results: list[AlignmentResult] = list()
    for prediction_labels in predictions_labels:
        json_labels = json.loads(Path(LABELED_DATA_PATH, "labels", prediction_labels.get_label_file_name()).read_text())
        truth_labels = Labels(**json_labels)
        precision, recall, f1_score = get_f1_score(truth_labels, prediction_labels)

        results.append(
            AlignmentResult(
                name=prediction_labels.get_label_file_name().rsplit(".", 1)[0],
                algorithm=model_name,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                total_paragraphs=len(truth_labels.paragraphs),
                seconds=prediction_labels.get_seconds(),
            )
        )

        if show_mistakes:
            save_mistakes(truth_labels, prediction_labels)

    results.append(get_average(results))
    markdown = markdown_table([x.model_dump() for x in results]).set_params(padding_width=5).get_markdown()
    print(markdown)


if __name__ == "__main__":
    model_name = "vgt_base"
    show_mistakes = True
    get_alignment_benchmark(model_name, show_mistakes)
