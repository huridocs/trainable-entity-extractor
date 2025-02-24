import pickle
from pathlib import Path

from multilingual_paragraph_extractor.driver.label_data import get_paths, EXTRACTION_IDENTIFIER, PARAGRAPH_EXTRACTION_PATH
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.SegmentationData import SegmentationData
from trainable_entity_extractor.use_cases.XmlFile import XmlFile

SEGMENTATION_DATA_PATH = Path(PARAGRAPH_EXTRACTION_PATH, "segmentation_data")


def save_pdfs_data():
    for xml_path in Path(PARAGRAPH_EXTRACTION_PATH, "xmls").iterdir():
        pdf_data_pickle = Path(PARAGRAPH_EXTRACTION_PATH, "pdf_data", xml_path.name.replace(".xml", ".pickle"))
        if pdf_data_pickle.exists():
            continue

        pdf_data = get_pdf_data(xml_path.name.replace(".xml", ""))

        with open(pdf_data_pickle, "wb") as f:
            pickle.dump(pdf_data, f)


def get_segmentation_data(pdf_path) -> SegmentationData:
    pdf_name = pdf_path.name.replace(".pdf", ".picke")
    segmentation_data_pickle = Path(SEGMENTATION_DATA_PATH, pdf_name)
    xml_segments_boxes = []
    if segmentation_data_pickle.exists():
        with open(segmentation_data_pickle, "rb") as f:
            segmentation_data = pickle.load(f)
            xml_segments_boxes = segmentation_data.xml_segments_boxes
    return SegmentationData(page_width=0, page_height=0, xml_segments_boxes=xml_segments_boxes)


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
