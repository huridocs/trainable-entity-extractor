import json
import pickle
from pathlib import Path
from pdf_token_type_labels.TokenType import TokenType

from multilingual_paragraph_extractor.driver.label_data import get_paths, EXTRACTION_IDENTIFIER, PARAGRAPH_EXTRACTION_PATH
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.SegmentBox import SegmentBox
from trainable_entity_extractor.domain.SegmentationData import SegmentationData
from trainable_entity_extractor.domain.XmlFile import XmlFileUseCase

SEGMENTATION_DATA_PATH = Path(PARAGRAPH_EXTRACTION_PATH, "segmentation_data")
DOCLING_JSONS_PATH = Path("/path/to/docling/jsons")

DOCLING_TYPE_TO_TOKEN_TYPE = {
    "caption": TokenType.CAPTION,
    "footnote": TokenType.FOOTNOTE,
    "formula": TokenType.FORMULA,
    "list_item": TokenType.LIST_ITEM,
    "page_footer": TokenType.PAGE_FOOTER,
    "page_header": TokenType.PAGE_HEADER,
    "picture": TokenType.PICTURE,
    "section_header": TokenType.SECTION_HEADER,
    "table": TokenType.TABLE,
    "text": TokenType.TEXT,
    "title": TokenType.TITLE,
    "document_index": TokenType.TABLE,
    "code": TokenType.TEXT,
    "checkbox_selected": TokenType.TEXT,
    "checkbox_unselected": TokenType.TEXT,
    "form": TokenType.TEXT,
    "key_value_region": TokenType.TEXT,
    "paragraph": TokenType.TEXT,
    "reference": TokenType.TEXT,
}


def save_pdfs_data():
    for xml_path in Path(PARAGRAPH_EXTRACTION_PATH, "xmls").iterdir():
        pdf_data_pickle = Path(PARAGRAPH_EXTRACTION_PATH, "pdf_data", xml_path.name.replace(".xml", ".pickle"))
        if pdf_data_pickle.exists():
            continue

        pdf_data = get_pdf_data(xml_path.name.replace(".xml", ""))

        with open(pdf_data_pickle, "wb") as f:
            pickle.dump(pdf_data, f)


def get_xml_segment_boxes(pdf_path: Path) -> list[SegmentBox]:
    xml_segment_boxes: list[SegmentBox] = []
    json_path = Path(DOCLING_JSONS_PATH, pdf_path.name.replace(".pdf", ".json"))
    json_data: dict = json.load(json_path.open())
    for page in json_data["pages"]:
        for label in page["page_labels"]:
            xml_segment_boxes.append(
                SegmentBox(
                    left=label["bounding_box"]["left"],
                    top=label["bounding_box"]["top"],
                    width=label["bounding_box"]["width"],
                    height=label["bounding_box"]["height"],
                    page_number=page["page_number"],
                    page_width=page["page_width"],
                    page_height=page["page_height"],
                    segment_type=TokenType(DOCLING_TYPE_TO_TOKEN_TYPE[label["segment_type"]]),
                )
            )
    return xml_segment_boxes


def get_segmentation_data(pdf_path) -> SegmentationData:
    xml_segments_boxes = get_xml_segment_boxes(pdf_path)
    return SegmentationData(page_width=0, page_height=0, xml_segments_boxes=xml_segments_boxes, label_segments_boxes=[])


def get_pdf_data(pdf_name: str):
    pdf_path, xml_path = get_paths(pdf_name)

    with open(xml_path, "rb") as file:
        xml_file = XmlFileUseCase(extraction_identifier=EXTRACTION_IDENTIFIER, to_train=True, xml_file_name=xml_path.name)
        xml_file.save(file_content=file.read())

    segmentation_data: SegmentationData = get_segmentation_data(pdf_path)
    pdf_data = PdfData.from_xml_file(xml_file=xml_file, segmentation_data=segmentation_data)
    return pdf_data


if __name__ == "__main__":
    save_pdfs_data()
