import json
import pickle
from enum import Enum
from pathlib import Path
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType
from pydantic import BaseModel, ConfigDict

from multilingual_paragraph_extractor.driver.label_data import get_paths, EXTRACTION_IDENTIFIER, PARAGRAPH_EXTRACTION_PATH
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.SegmentBox import SegmentBox
from trainable_entity_extractor.domain.SegmentationData import SegmentationData
from trainable_entity_extractor.use_cases.XmlFile import XmlFile

SEGMENTATION_DATA_PATH = Path(PARAGRAPH_EXTRACTION_PATH, "segmentation_data")
DOCLING_JSONS_PATH = Path("docling/jsons/path")


class DocItemLabel(str, Enum):
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    FORMULA = "formula"
    LIST_ITEM = "list_item"
    PAGE_FOOTER = "page_footer"
    PAGE_HEADER = "page_header"
    PICTURE = "picture"
    SECTION_HEADER = "section_header"
    TABLE = "table"
    TEXT = "text"
    TITLE = "title"
    DOCUMENT_INDEX = "document_index"
    CODE = "code"
    CHECKBOX_SELECTED = "checkbox_selected"
    CHECKBOX_UNSELECTED = "checkbox_unselected"
    FORM = "form"
    KEY_VALUE_REGION = "key_value_region"

    # Additional labels for markup-based formats (e.g. HTML, Word)
    PARAGRAPH = "paragraph"
    REFERENCE = "reference"

    def __str__(self):
        return str(self.value)


DOCLING_TYPE_TO_TOKEN_TYPE = {
    DocItemLabel.CAPTION: TokenType.CAPTION,
    DocItemLabel.FOOTNOTE: TokenType.FOOTNOTE,
    DocItemLabel.FORMULA: TokenType.FORMULA,
    DocItemLabel.LIST_ITEM: TokenType.LIST_ITEM,
    DocItemLabel.PAGE_FOOTER: TokenType.PAGE_FOOTER,
    DocItemLabel.PAGE_HEADER: TokenType.PAGE_HEADER,
    DocItemLabel.PICTURE: TokenType.PICTURE,
    DocItemLabel.SECTION_HEADER: TokenType.SECTION_HEADER,
    DocItemLabel.TABLE: TokenType.TABLE,
    DocItemLabel.TEXT: TokenType.TEXT,
    DocItemLabel.TITLE: TokenType.TITLE,
    DocItemLabel.DOCUMENT_INDEX: TokenType.TABLE,
    DocItemLabel.CODE: TokenType.TEXT,
    DocItemLabel.CHECKBOX_SELECTED: TokenType.TEXT,
    DocItemLabel.CHECKBOX_UNSELECTED: TokenType.TEXT,
    DocItemLabel.FORM: TokenType.TEXT,
    DocItemLabel.KEY_VALUE_REGION: TokenType.TEXT,
    DocItemLabel.PARAGRAPH: TokenType.TEXT,
    DocItemLabel.REFERENCE: TokenType.TEXT,
}


class DoclingLabel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __hash__(self):
        return hash((self.text, self.segment_type, self.bounding_box))

    text: str
    segment_type: DocItemLabel
    bounding_box: Rectangle

    @staticmethod
    def bbox_to_rectangle(bbox: dict, page_height: int) -> Rectangle:
        left = bbox["l"]
        top = page_height - bbox["t"]
        right = bbox["r"]
        bottom = page_height - bbox["b"]
        return Rectangle.from_coordinates(left=left, top=top, right=right, bottom=bottom)

    @staticmethod
    def from_label_json(label_json: dict, page_height: int) -> "DoclingLabel":
        segment_type = DocItemLabel(label_json["label"])
        text = label_json["text"]
        bounding_box = DoclingLabel.bbox_to_rectangle(label_json["prov"][0]["bbox"], page_height)
        return DoclingLabel(text=text, segment_type=segment_type, bounding_box=bounding_box)


class DoclingPage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    page_number: int
    page_width: int
    page_height: int
    page_labels: list[DoclingLabel]

    @staticmethod
    def from_page_content(page_number: int, width: int, height: int, page_content: list) -> "DoclingPage":
        page_labels: list[DoclingLabel] = [DoclingLabel.from_label_json(label, height) for label in page_content]
        return DoclingPage(page_number=page_number, page_width=width, page_height=height, page_labels=page_labels)


class DoclingDocument(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    json_path: Path
    pages: list[DoclingPage]

    @staticmethod
    def _process_media_item(media_item: dict, texts_by_refs: dict, media_type: str) -> list[dict]:
        result = []
        captions = [texts_by_refs[ref["$ref"]] for ref in media_item["captions"]]
        if media_type == "picture":
            picture_text_refs = [
                texts_by_refs[ref["$ref"]] for ref in media_item["children"] if texts_by_refs[ref["$ref"]] not in captions
            ]
            media_item["text"] = " ".join([ref["text"] for ref in picture_text_refs])

        media_top = media_item["prov"][0]["bbox"]["t"]

        # coordinate system is different, so we use > instead of <
        captions_before = [caption for caption in captions if caption["prov"][0]["bbox"]["t"] > media_top]
        captions_after = [caption for caption in captions if caption not in captions_before]

        result.extend(captions_before)
        result.append(media_item)
        result.extend(captions_after)

        return result

    @staticmethod
    def get_document_content(json_data: dict) -> list[dict]:
        reference_ids_in_body = [item["$ref"] for item in json_data["body"]["children"]]
        texts_by_refs = {text["self_ref"]: text for text in json_data["texts"]}
        pictures_by_refs = {picture["self_ref"]: picture for picture in json_data["pictures"]}
        tables_by_refs = {table["self_ref"]: table for table in json_data["tables"]}
        groups_by_refs = {group["self_ref"]: group for group in json_data["groups"]}

        document_content = []

        for reference_id in reference_ids_in_body:
            if "text" in reference_id:
                document_content.append(texts_by_refs[reference_id])
            elif "pictures" in reference_id:
                picture = pictures_by_refs[reference_id]
                document_content.extend(
                    DoclingDocument._process_media_item(
                        media_item=picture, texts_by_refs=texts_by_refs, media_type="picture"
                    )
                )
            elif "tables" in reference_id:
                table = tables_by_refs[reference_id]
                table["text"] = " ".join(cell["text"] for cell in table["data"]["table_cells"])
                document_content.extend(
                    DoclingDocument._process_media_item(media_item=table, texts_by_refs=texts_by_refs, media_type="table")
                )
            elif "groups" in reference_id:
                group = groups_by_refs[reference_id]
                text_reference_ids_in_group = [item["$ref"] for item in group["children"]]
                for text_reference_id in text_reference_ids_in_group:
                    document_content.append(texts_by_refs[text_reference_id])

        return document_content

    @staticmethod
    def from_json_path(json_path: Path) -> "DoclingDocument":
        json_data: dict = json.load(json_path.open())
        document_content = DoclingDocument.get_document_content(json_data)

        page_content_by_page_number: dict[int, list] = {}
        for item in document_content:
            page_content_by_page_number.setdefault(item["prov"][0]["page_no"], []).append(item)
        page_numbers = sorted(list(page_content_by_page_number.keys()))
        pages: list[DoclingPage] = []
        for page_number in page_numbers:
            page_width = round(json_data["pages"][str(page_number)]["size"]["width"])
            page_height = round(json_data["pages"][str(page_number)]["size"]["height"])
            page_content = page_content_by_page_number[page_number]
            docling_page = DoclingPage.from_page_content(page_number, page_width, page_height, page_content)
            pages.append(docling_page)
        return DoclingDocument(json_path=json_path, pages=pages)


def save_pdfs_data():
    for xml_path in Path(PARAGRAPH_EXTRACTION_PATH, "xmls").iterdir():
        pdf_data_pickle = Path(PARAGRAPH_EXTRACTION_PATH, "pdf_data", xml_path.name.replace(".xml", ".pickle"))
        if pdf_data_pickle.exists():
            continue

        pdf_data = get_pdf_data(xml_path.name.replace(".xml", ""))

        with open(pdf_data_pickle, "wb") as f:
            pickle.dump(pdf_data, f)


def get_xml_segment_boxes(docling_document: DoclingDocument):
    xml_segment_boxes: list[SegmentBox] = []
    for page in docling_document.pages:
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
                    segment_type=DOCLING_TYPE_TO_TOKEN_TYPE[label.segment_type],
                )
            )
    return xml_segment_boxes


def get_segmentation_data(pdf_path) -> SegmentationData:
    pdf_name = pdf_path.name.replace(".pdf", "")
    json_path = Path(DOCLING_JSONS_PATH, pdf_name + ".json")
    docling_document: DoclingDocument = DoclingDocument.from_json_path(json_path=json_path)

    xml_segments_boxes = get_xml_segment_boxes(docling_document)
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
