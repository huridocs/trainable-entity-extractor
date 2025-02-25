import json
import pickle
from enum import StrEnum
from pathlib import Path

from bs4 import BeautifulSoup
from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.PdfToken import PdfToken
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType
from pydantic import BaseModel, ConfigDict

from multilingual_paragraph_extractor.driver.label_data import get_paths, EXTRACTION_IDENTIFIER, PARAGRAPH_EXTRACTION_PATH
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.SegmentBox import SegmentBox
from trainable_entity_extractor.domain.SegmentationData import SegmentationData
from trainable_entity_extractor.use_cases.XmlFile import XmlFile

SEGMENTATION_DATA_PATH = Path(PARAGRAPH_EXTRACTION_PATH, "segmentation_data")
MARKER_JSONS_PATH = Path("marker/jsons/path")


class MarkerTokenType(StrEnum):
    Line = "Line"
    Span = "Span"
    FigureGroup = "FigureGroup"
    TableGroup = "TableGroup"
    ListGroup = "ListGroup"
    PictureGroup = "PictureGroup"
    Page = "Page"
    Caption = "Caption"
    Code = "Code"
    Figure = "Figure"
    Footnote = "Footnote"
    Form = "Form"
    Equation = "Equation"
    Handwriting = "Handwriting"
    TextInlineMath = "TextInlineMath"
    ListItem = "ListItem"
    PageFooter = "PageFooter"
    PageHeader = "PageHeader"
    Picture = "Picture"
    SectionHeader = "SectionHeader"
    Table = "Table"
    Text = "Text"
    TableOfContents = "TableOfContents"
    Document = "Document"
    ComplexRegion = "ComplexRegion"
    TableCell = "TableCell"
    Reference = "Reference"


MARKER_TYPE_TO_TOKEN_TYPE = {
    MarkerTokenType.Line: TokenType.TEXT,
    MarkerTokenType.Span: TokenType.TEXT,
    MarkerTokenType.FigureGroup: TokenType.PICTURE,
    MarkerTokenType.TableGroup: TokenType.TABLE,
    MarkerTokenType.ListGroup: TokenType.LIST_ITEM,
    MarkerTokenType.PictureGroup: TokenType.PICTURE,
    MarkerTokenType.Page: TokenType.TEXT,
    MarkerTokenType.Caption: TokenType.CAPTION,
    MarkerTokenType.Code: TokenType.TEXT,
    MarkerTokenType.Figure: TokenType.PICTURE,
    MarkerTokenType.Footnote: TokenType.FOOTNOTE,
    MarkerTokenType.Form: TokenType.TEXT,
    MarkerTokenType.Equation: TokenType.FORMULA,
    MarkerTokenType.Handwriting: TokenType.TEXT,
    MarkerTokenType.TextInlineMath: TokenType.TEXT,
    MarkerTokenType.ListItem: TokenType.LIST_ITEM,
    MarkerTokenType.PageFooter: TokenType.PAGE_FOOTER,
    MarkerTokenType.PageHeader: TokenType.PAGE_HEADER,
    MarkerTokenType.Picture: TokenType.PICTURE,
    MarkerTokenType.SectionHeader: TokenType.SECTION_HEADER,
    MarkerTokenType.Table: TokenType.TABLE,
    MarkerTokenType.Text: TokenType.TEXT,
    MarkerTokenType.TableOfContents: TokenType.TABLE,
    MarkerTokenType.Document: TokenType.TEXT,
    MarkerTokenType.ComplexRegion: TokenType.TEXT,
    MarkerTokenType.TableCell: TokenType.TABLE,
    MarkerTokenType.Reference: TokenType.TEXT,
}


class MarkerLabel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __hash__(self):
        return hash((self.text, self.html, self.segment_type, self.bounding_box))

    text: str
    html: str
    segment_type: MarkerTokenType
    bounding_box: Rectangle

    @staticmethod
    def polygon_to_rectangle(polygon: list[list[float], list[float], list[float], list[float]]) -> Rectangle:
        x_min = int(polygon[0][0])
        y_min = int(polygon[0][1])
        x_max = int(polygon[1][0])
        y_max = int(polygon[2][1])
        return Rectangle.from_coordinates(left=x_min, top=y_min, right=x_max, bottom=y_max)

    @staticmethod
    def extract_html_content(html_string: str):
        soup = BeautifulSoup(html_string, "html.parser")
        clean_text = soup.get_text()
        return clean_text

    @staticmethod
    def from_label_json(label_json: dict) -> "MarkerLabel":
        segment_type = MarkerTokenType(label_json["block_type"])
        html = label_json["html"]
        text = MarkerLabel.extract_html_content(label_json["html"])
        bounding_box = MarkerLabel.polygon_to_rectangle(label_json["polygon"])
        return MarkerLabel(text=text, html=html, segment_type=segment_type, bounding_box=bounding_box)


class MarkerPage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    page_number: int
    page_width: int
    page_height: int
    page_labels: list[MarkerLabel]

    @staticmethod
    def extract_page_number(page_id: str) -> int:
        try:
            return int(page_id.split("/")[2]) + 1
        except (IndexError, ValueError):
            return 0

    @staticmethod
    def from_page_json(page_json: dict) -> "MarkerPage":
        page_number = MarkerPage.extract_page_number(page_json["id"])
        width = int(page_json["bbox"][2])
        height = int(page_json["bbox"][3])
        page_labels: list[MarkerLabel] = [MarkerLabel.from_label_json(label) for label in page_json["children"]]
        return MarkerPage(page_number=page_number, page_width=width, page_height=height, page_labels=page_labels)


class MarkerDocument(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    json_path: Path
    pages: list[MarkerPage]

    @staticmethod
    def from_json_path(json_path: Path) -> "MarkerDocument":
        json_data: dict = json.load(json_path.open())
        pages: list[MarkerPage] = [MarkerPage.from_page_json(page_json=page_json) for page_json in json_data["children"]]
        return MarkerDocument(json_path=json_path, pages=pages)


def save_pdfs_data():
    for xml_path in Path(PARAGRAPH_EXTRACTION_PATH, "xmls").iterdir():
        pdf_data_pickle = Path(PARAGRAPH_EXTRACTION_PATH, "pdf_data", xml_path.name.replace(".xml", ".pickle"))
        if pdf_data_pickle.exists():
            continue

        pdf_data = get_pdf_data(xml_path.name.replace(".xml", ""))

        with open(pdf_data_pickle, "wb") as f:
            pickle.dump(pdf_data, f)


def get_xml_segment_boxes(marker_document: MarkerDocument):
    xml_segment_boxes: list[SegmentBox] = []
    for page in marker_document.pages:
        for label in page.page_labels:
            xml_segment_boxes.append(
                SegmentBox(
                    left=label.bounding_box.left,
                    top=label.bounding_box.top,
                    width=label.bounding_box.width,
                    height=label.bounding_box.height,
                    page_number=page.page_number,
                    page_width=page.page_width,
                    page_height=page.page_height,
                    segment_type=MARKER_TYPE_TO_TOKEN_TYPE[label.segment_type],
                )
            )
    return xml_segment_boxes


def remove_no_token_marker_labels(marker_labels: list[MarkerLabel], pdf_tokens: list[PdfToken]):
    labels_to_keep = []
    for pdf_token in pdf_tokens:
        for marker_label in marker_labels:
            if marker_label.bounding_box.get_intersection_percentage(pdf_token.bounding_box):
                labels_to_keep.append(marker_label)
                break
    return [marker_label for marker_label in marker_labels if marker_label in labels_to_keep]


def get_segmentation_data(pdf_path) -> SegmentationData:
    pdf_name = pdf_path.name.replace(".pdf", "")
    json_path = Path(MARKER_JSONS_PATH, pdf_name, pdf_name + ".json")
    marker_document: MarkerDocument = MarkerDocument.from_json_path(json_path=json_path)
    pdf_features = PdfFeatures.from_pdf_path(pdf_path=pdf_path)
    for page in pdf_features.pages:
        marker_page = [p for p in marker_document.pages if p.page_number == page.page_number][0]
        marker_page.page_labels = remove_no_token_marker_labels(marker_page.page_labels, page.tokens)

    xml_segments_boxes = get_xml_segment_boxes(marker_document)
    return SegmentationData(page_width=0, page_height=0, xml_segments_boxes=xml_segments_boxes, label_segments_boxes=[])


def get_pdf_data(pdf_name: str):
    pdf_path, xml_path = get_paths(pdf_name)

    with open(xml_path, "rb") as file:
        xml_file = XmlFile(extraction_identifier=EXTRACTION_IDENTIFIER, to_train=True, xml_file_name=xml_path.name)
        xml_file.save(file_content=file.read())

    segmentation_data: SegmentationData = get_segmentation_data(pdf_path)
    pdf_data = PdfData.from_xml_file(xml_file=xml_file, segmentation_data=segmentation_data)
    return pdf_data


if __name__ == "__main__":
    save_pdfs_data()
