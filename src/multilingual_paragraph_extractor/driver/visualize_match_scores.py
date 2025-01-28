import json
import pickle
import subprocess
from pathlib import Path

from pdf_annotate import PdfAnnotator, Appearance, Location
from pdf_token_type_labels.TokenType import TokenType
from visualization.save_output_to_pdf import hex_color_to_rgb

from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures
from multilingual_paragraph_extractor.domain.ParagraphMatchScore import ParagraphMatchScore
from trainable_entity_extractor.XmlFile import XmlFile
from trainable_entity_extractor.config import ROOT_PATH
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.PdfData import PdfData
from trainable_entity_extractor.data.SegmentBox import SegmentBox
from trainable_entity_extractor.data.SegmentationData import SegmentationData

PARAGRAPH_EXTRACTION_PATH = Path(ROOT_PATH, "data", "paragraph_extraction")
EXTRACTION_IDENTIFIER = ExtractionIdentifier(run_name="paragraph", extraction_name="id")

MAIN_PDF = "ihrda_1_en"
OTHER_PDF = "ihrda_1_fr"


def get_pdf_data(pdf_name: str):
    pdf_path, xml_path = get_paths(pdf_name)

    with open(xml_path, "rb") as file:
        xml_file = XmlFile(extraction_identifier=EXTRACTION_IDENTIFIER, to_train=True, xml_file_name=xml_path.name)
        xml_file.save(file_content=file.read())

    segmentation_data = get_segmentation_data(pdf_path)
    pdf_data = PdfData.from_xml_file(xml_file=xml_file, segmentation_data=segmentation_data)
    return pdf_data


def get_paths(pdf_name):
    xml_name = pdf_name + ".xml"
    pdf_name = pdf_name + ".pdf"
    xml_path = Path(PARAGRAPH_EXTRACTION_PATH, "xmls", xml_name)
    pdf_path = Path(PARAGRAPH_EXTRACTION_PATH, "pdfs", pdf_name)
    return pdf_path, xml_path


def save_xmls():
    for pdf_path in Path(PARAGRAPH_EXTRACTION_PATH, "pdfs").iterdir():
        xml_path = Path(PARAGRAPH_EXTRACTION_PATH, "xmls", pdf_path.name.replace(".pdf", ".xml"))
        if xml_path.exists():
            continue
        subprocess.run(["pdftohtml", "-i", "-xml", "-zoom", "1.0", pdf_path, xml_path])


def save_pdfs_data():
    for xml_path in Path(PARAGRAPH_EXTRACTION_PATH, "xmls").iterdir():
        if "1_" not in xml_path.name:
            continue
        pdf_data_pickle = Path(PARAGRAPH_EXTRACTION_PATH, "pdf_data", xml_path.name.replace(".xml", ".pickle"))
        if pdf_data_pickle.exists():
            continue

        pdf_data = get_pdf_data(xml_path.name.replace(".xml", ""))

        with open(pdf_data_pickle, "wb") as f:
            pickle.dump(pdf_data, f)


def get_segmentation_data(pdf_path: Path):
    command = [
        "curl",
        "-X",
        "POST",
        "-F",
        f"file=@{pdf_path}",
        "localhost:5060",
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    json_data = json.loads(result.stdout)

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


def load_pdf_data(pdf_name: str) -> PdfData:
    pdf_data_pickle = Path(PARAGRAPH_EXTRACTION_PATH, "pdf_data", pdf_name + ".pickle")
    with open(pdf_data_pickle, "rb") as file:
        return pickle.load(file)


def get_score(paragraph_1: ParagraphFeatures, paragraph_2: ParagraphFeatures):
    return paragraph_1, paragraph_2, ParagraphMatchScore.from_paragraphs_features(paragraph_1, paragraph_2).overall_score


def annotate_pdf(scores, scores_next, scores_next_next):
    pdf_path, xml_path = get_paths(MAIN_PDF)
    annotator = PdfAnnotator(str(pdf_path))
    for score, score_next, score_next_next in zip(scores, scores_next, scores_next_next):
        paragraph_1, paragraph_2, match_score = score
        _, paragraph_2_next, match_score_next = score_next
        _, paragraph_2_next_next, match_score_next_next = score_next_next

        left, top, right, bottom = (
            paragraph_1.bounding_box.left,
            paragraph_1.page_height - paragraph_1.bounding_box.top,
            paragraph_1.bounding_box.right,
            paragraph_1.page_height - paragraph_1.bounding_box.bottom,
        )

        text_box_size = 20 * 8 + 8

        annotator.add_annotation(
            "square",
            Location(x1=left, y1=bottom, x2=right, y2=top, page=paragraph_1.page_number - 1),
            Appearance(stroke_color=hex_color_to_rgb("#008B8B")),
        )

        annotator.add_annotation(
            "square",
            Location(x1=left, y1=top, x2=left + text_box_size, y2=top + 10, page=paragraph_1.page_number - 1),
            Appearance(fill=hex_color_to_rgb("#008B8B")),
        )

        content = f"{paragraph_2.first_word} {int(100 * match_score)} | "
        content += f"{paragraph_2_next.first_word} {int(100 * match_score_next)} | "
        content += f"{paragraph_2_next_next.first_word} {int(100 * match_score_next_next)}"

        annotator.add_annotation(
            "text",
            Location(x1=left, y1=top, x2=left + text_box_size, y2=top + 10, page=paragraph_1.page_number - 1),
            Appearance(content=content, font_size=8, fill=(1, 1, 1), stroke_width=3),
        )

    output_pdf_path = Path(PARAGRAPH_EXTRACTION_PATH, MAIN_PDF + ".pdf")
    annotator.write(output_pdf_path)


def get_scores(main_paragraphs_features, other_paragraphs_features):
    scores = [get_score(x, y) for x, y in zip(main_paragraphs_features[1:], other_paragraphs_features)]
    scores_next = [get_score(x, y) for x, y in zip(main_paragraphs_features[1:], other_paragraphs_features[1:])]
    scores_next_next = [get_score(x, y) for x, y in zip(main_paragraphs_features[1:], other_paragraphs_features[2:])]
    return scores, scores_next, scores_next_next


def get_paragraphs():
    main_pdf = load_pdf_data(MAIN_PDF)
    other_pdf = load_pdf_data(OTHER_PDF)
    main_paragraphs_features = [ParagraphFeatures.from_pdf_data(main_pdf, x) for x in main_pdf.pdf_data_segments]
    other_paragraphs_features = [ParagraphFeatures.from_pdf_data(other_pdf, x) for x in other_pdf.pdf_data_segments]
    text_content_types = [
        TokenType.FORMULA,
        TokenType.LIST_ITEM,
        TokenType.TITLE,
        TokenType.TEXT,
        TokenType.SECTION_HEADER,
        TokenType.TABLE,
    ]
    main_paragraphs_features = [x for x in main_paragraphs_features if x.segment_type in text_content_types]
    other_paragraphs_features = [x for x in other_paragraphs_features if x.segment_type in text_content_types]
    return main_paragraphs_features, other_paragraphs_features


def visualize():
    save_xmls()
    save_pdfs_data()
    main_paragraphs_features, other_paragraphs_features = get_paragraphs()
    scores, scores_next, scores_next_next = get_scores(main_paragraphs_features, other_paragraphs_features)
    annotate_pdf(scores, scores_next, scores_next_next)
    print("ok")


if __name__ == "__main__":
    visualize()
