import itertools
import json
import pickle
import subprocess
from pathlib import Path
from time import time

from pdf_annotate import Location, Appearance, PdfAnnotator
from pdf_token_type_labels.TokenType import TokenType
from visualization.save_output_to_pdf import hex_color_to_rgb

from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures
from multilingual_paragraph_extractor.domain.ParagraphMatchScore import ParagraphMatchScore
from multilingual_paragraph_extractor.domain.ParagraphsFromLanguage import ParagraphsFromLanguage
from multilingual_paragraph_extractor.driver.Labels import Labels
from multilingual_paragraph_extractor.use_cases.MultilingualParagraphAlignerUseCase import (
    MultilingualParagraphAlignerUseCase,
)
from trainable_entity_extractor.XmlFile import XmlFile
from trainable_entity_extractor.config import ROOT_PATH, APP_PATH
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.PdfData import PdfData
from trainable_entity_extractor.data.SegmentBox import SegmentBox
from trainable_entity_extractor.data.SegmentationData import SegmentationData

PARAGRAPH_EXTRACTION_PATH = Path(ROOT_PATH, "data", "paragraph_extraction")
EXTRACTION_IDENTIFIER = ExtractionIdentifier(run_name="paragraph", extraction_name="id")
LABELED_DATA_PATH = Path(APP_PATH, "multilingual_paragraph_extractor", "resources", "labeled_data")


def save_xmls():
    for pdf_path in Path(LABELED_DATA_PATH, "pdfs").iterdir():
        xml_path = Path(PARAGRAPH_EXTRACTION_PATH, "xmls", pdf_path.name.replace(".pdf", ".xml"))
        if xml_path.exists():
            continue
        subprocess.run(["pdftohtml", "-i", "-xml", "-zoom", "1.0", pdf_path, xml_path])


def get_paths(pdf_name):
    xml_name = pdf_name + ".xml"
    pdf_name = pdf_name + ".pdf"
    xml_path = Path(PARAGRAPH_EXTRACTION_PATH, "xmls", xml_name)
    pdf_path = Path(LABELED_DATA_PATH, "pdfs", pdf_name)
    return pdf_path, xml_path


def get_segmentation_data(pdf_path: Path):
    command = [
        "curl",
        "-X",
        "POST",
        "-F",
        f"file=@{pdf_path}",
        "localhost:5060",
    ]

    result = subprocess.run(command, capture_output=True, text=False)
    json_data = json.loads(result.stdout.decode("utf-8"))

    for data in json_data:
        data["segment_type"] = data["type"]

    segment_boxes = [SegmentBox(**segment) for segment in json_data]

    segmentation_data = SegmentationData(
        page_width=segment_boxes[0].page_width,
        page_height=segment_boxes[0].page_height,
        xml_segments_boxes=segment_boxes,
        label_segments_boxes=[],
    )

    return segmentation_data


def get_pdf_data(pdf_name: str):
    pdf_path, xml_path = get_paths(pdf_name)

    with open(xml_path, "rb") as file:
        xml_file = XmlFile(extraction_identifier=EXTRACTION_IDENTIFIER, to_train=True, xml_file_name=xml_path.name)
        xml_file.save(file_content=file.read())

    segmentation_data = get_segmentation_data(pdf_path)
    pdf_data = PdfData.from_xml_file(xml_file=xml_file, segmentation_data=segmentation_data)
    return pdf_data


def save_pdfs_data():
    for xml_path in Path(PARAGRAPH_EXTRACTION_PATH, "xmls").iterdir():
        pdf_data_pickle = Path(PARAGRAPH_EXTRACTION_PATH, "pdf_data", xml_path.name.replace(".xml", ".pickle"))
        if pdf_data_pickle.exists():
            continue

        pdf_data = get_pdf_data(xml_path.name.replace(".xml", ""))

        with open(pdf_data_pickle, "wb") as f:
            pickle.dump(pdf_data, f)


def load_pdf_data(pdf_name: str) -> PdfData:
    pdf_data_pickle = Path(PARAGRAPH_EXTRACTION_PATH, "pdf_data", pdf_name + ".pickle")
    with open(pdf_data_pickle, "rb") as file:
        return pickle.load(file)


def get_paragraphs(pdf_name: str):
    pdf_data = load_pdf_data(pdf_name)
    paragraphs_features = [ParagraphFeatures.from_pdf_data(pdf_data, x) for x in pdf_data.pdf_data_segments]
    text_content_types = [
        TokenType.FORMULA,
        TokenType.LIST_ITEM,
        TokenType.TITLE,
        TokenType.TEXT,
        TokenType.SECTION_HEADER,
        TokenType.TABLE,
    ]
    paragraphs_features = [x for x in paragraphs_features if x.paragraph_type in text_content_types]
    return paragraphs_features


def loop_combinations() -> tuple[str, ParagraphsFromLanguage, ParagraphsFromLanguage]:

    pdf_languages: dict[str, set[str]] = {}

    for pdf_path in Path(LABELED_DATA_PATH, "pdfs").iterdir():
        if pdf_path.name.endswith(".pdf"):
            base_name, language = pdf_path.name.rsplit("_", 1)
            pdf_languages.setdefault(base_name, set()).add(language[:2])

    for pdf_name, languages in pdf_languages.items():
        if len(languages) < 2:
            continue

        for main_language, other_language in itertools.permutations(languages, 2):
            main_paragraphs = ParagraphsFromLanguage(
                language=main_language, paragraphs=get_paragraphs(f"{pdf_name}_{main_language}"), is_main_language=True
            )
            other_paragraphs = ParagraphsFromLanguage(
                language=other_language, paragraphs=get_paragraphs(f"{pdf_name}_{other_language}"), is_main_language=False
            )
            yield pdf_name, main_paragraphs, other_paragraphs


def annotate_pdf(pdf_name: str, output_name: str, paragraphs: list[ParagraphFeatures], contents: list[str]):
    pdf_path, xml_path = get_paths(pdf_name)
    annotator = PdfAnnotator(str(pdf_path))
    for paragraph, content in zip(paragraphs, contents):
        left, top, right, bottom = (
            paragraph.bounding_box.left,
            paragraph.page_height - paragraph.bounding_box.top,
            paragraph.bounding_box.right,
            paragraph.page_height - paragraph.bounding_box.bottom,
        )

        text_box_size = 20 * 8 + 8

        annotator.add_annotation(
            "square",
            Location(x1=left, y1=bottom, x2=right, y2=top, page=paragraph.page_number - 1),
            Appearance(stroke_color=hex_color_to_rgb("#008B8B")),
        )

        annotator.add_annotation(
            "square",
            Location(x1=left, y1=top, x2=left + text_box_size, y2=top + 10, page=paragraph.page_number - 1),
            Appearance(fill=hex_color_to_rgb("#008B8B")),
        )

        annotator.add_annotation(
            "text",
            Location(x1=left, y1=top, x2=left + text_box_size, y2=top + 10, page=paragraph.page_number - 1),
            Appearance(content=content, font_size=8, fill=(1, 1, 1), stroke_width=3),
        )

    output_pdf_path = Path(PARAGRAPH_EXTRACTION_PATH, output_name + ".pdf")
    annotator.write(output_pdf_path)


def get_score(paragraph_1: ParagraphFeatures, paragraph_2: ParagraphFeatures):
    return paragraph_1, paragraph_2, ParagraphMatchScore.from_paragraphs_features(paragraph_1, paragraph_2).overall_score


def get_scores(main_paragraphs_features, other_paragraphs_features):
    scores = [get_score(x, y) for x, y in zip(main_paragraphs_features[1:], other_paragraphs_features)]
    scores_next = [get_score(x, y) for x, y in zip(main_paragraphs_features[1:], other_paragraphs_features[1:])]
    scores_next_next = [get_score(x, y) for x, y in zip(main_paragraphs_features[1:], other_paragraphs_features[2:])]
    return scores, scores_next, scores_next_next


def visualize_matching_scores():
    for pdf_name, main_paragraphs, other_paragraphs in loop_combinations():
        scores, scores_next, scores_next_next = get_scores(main_paragraphs.paragraphs, other_paragraphs.paragraphs)
        output_name = f"scores_{pdf_name}_{main_paragraphs.language}_{other_paragraphs.language}"
        input_name = f"{pdf_name}_{main_paragraphs.language}"

        paragraphs = list()
        contents = list()
        for score, score_next, score_next_next in zip(scores, scores_next, scores_next_next):
            paragraph_1, paragraph_2, match_score = score
            _, paragraph_2_next, match_score_next = score_next
            _, paragraph_2_next_next, match_score_next_next = score_next_next
            paragraphs.append(paragraph_1)
            contents.append(
                f"{paragraph_2.first_word} {int(100 * match_score)} | "
                f"{paragraph_2_next.first_word} {int(100 * match_score_next)} | "
                f"{paragraph_2_next_next.first_word} {int(100 * match_score_next_next)}"
            )

        annotate_pdf(input_name, output_name, paragraphs, contents)

    print("ok")


def visualize_alignment():
    for pdf_name, main_paragraphs, other_paragraphs in loop_combinations():
        MultilingualParagraphAlignerUseCase(EXTRACTION_IDENTIFIER).align_languages([main_paragraphs, other_paragraphs])
        contents = list()
        for paragraph in main_paragraphs.paragraphs:
            if paragraph not in other_paragraphs._alignment_scores:
                contents.append("NO TRANSLATION")
                continue
            alignment_score = other_paragraphs._alignment_scores[paragraph]
            content = f"{alignment_score.other_paragraph.first_word} {int(100 * alignment_score.score)}"
            contents.append(content)

        output_name = f"alignment_{pdf_name}_{main_paragraphs.language}_{other_paragraphs.language}"
        input_name = f"{pdf_name}_{main_paragraphs.language}"
        annotate_pdf(input_name, output_name, main_paragraphs.paragraphs, contents)


def get_algorithm_labels():
    labels_list: list[Labels] = list()
    times = list()
    for pdf_name, main_paragraphs, other_paragraphs in loop_combinations():
        label_file_name = pdf_name + "_" + main_paragraphs.language + "_" + other_paragraphs.language + ".json"
        output_path = Path(PARAGRAPH_EXTRACTION_PATH, "labels", label_file_name)

        start = time()
        MultilingualParagraphAlignerUseCase(EXTRACTION_IDENTIFIER).align_languages([main_paragraphs, other_paragraphs])
        times.append(round(time() - start, 2))
        label = Labels(
            main_language=main_paragraphs.language,
            other_language=other_paragraphs.language,
            main_xml_name=pdf_name + "_" + main_paragraphs.language + ".xml",
            other_xml_name=pdf_name + "_" + other_paragraphs.language + ".xml",
            paragraphs=[],
        )
        contents = list()
        for paragraph in main_paragraphs.paragraphs:
            if paragraph not in other_paragraphs._alignment_scores:
                contents.append("NO TRANSLATION")
                continue
            alignment_score = other_paragraphs._alignment_scores[paragraph]
            label.add_paragraph(paragraph.original_text, alignment_score.other_paragraph.original_text)

        labels_list.append(label)
        output_path.write_text(json.dumps(label.model_dump(), indent=4, ensure_ascii=False), encoding="utf-8")
    return labels_list, times


if __name__ == "__main__":
    save_xmls()
    save_pdfs_data()
    # visualize_matching_scores()
    # visualize_alignment()
    # get_algorithm_labels()
