from typing import Optional

from trainable_entity_extractor.data.SegmentationData import SegmentationData
from pdf_features.PdfFeatures import PdfFeatures

from trainable_entity_extractor.FilterValidSegmentsPages import FilterValidSegmentsPages
from trainable_entity_extractor.data.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.XmlFile import XmlFile


class PdfData:
    def __init__(self, pdf_features: Optional[PdfFeatures], file_name="", file_type: str = ""):
        self.pdf_features: PdfFeatures = pdf_features
        if not file_name and pdf_features:
            self.file_name = pdf_features.file_name
        else:
            self.file_name = file_name
        self.file_type = file_type
        self.pdf_path = ""
        self.pdf_data_segments: list[PdfDataSegment] = list()

    def set_segments_from_segmentation_data(self, segmentation_data: SegmentationData):
        pdf_segments_to_merge = dict()
        pdf_segments_from_segmentation = [
            segment_box.to_pdf_segment() for segment_box in segmentation_data.xml_segments_boxes
        ]
        for page, token in self.pdf_features.loop_tokens():
            segment_from_tag: PdfDataSegment = PdfDataSegment.from_pdf_token(token)

            intersects_segmentation = [
                segmentation_segment
                for segmentation_segment in pdf_segments_from_segmentation
                if segmentation_segment.intersects(segment_from_tag)
            ]

            if not intersects_segmentation:
                self.pdf_data_segments.append(segment_from_tag)
                continue

            segment_from_tag.segment_type = intersects_segmentation[0].segment_type
            pdf_segments_to_merge.setdefault(intersects_segmentation[0], []).append(segment_from_tag)

        self.pdf_data_segments.extend(
            [
                PdfDataSegment.from_list_to_merge(each_pdf_segments_to_merge)
                for each_pdf_segments_to_merge in pdf_segments_to_merge.values()
            ]
        )
        self.pdf_data_segments.sort(key=lambda x: (x.page_number, x.bounding_box.top, x.bounding_box.left))

    def set_ml_label_from_segmentation_data(self, segmentation_data: SegmentationData):
        for label_segment_box in segmentation_data.label_segments_boxes:
            for segment in self.pdf_data_segments:
                if segment.page_number != label_segment_box.page_number:
                    continue
                if segment.is_selected(label_segment_box.get_bounding_box()):
                    segment.ml_label = 1

    def clean_text(self):
        for segment in self.pdf_data_segments:
            segment.text_content = " ".join(segment.text_content.split())

    @staticmethod
    def get_blank():
        return PdfData(None)

    @staticmethod
    def from_xml_file(xml_file: XmlFile, segmentation_data: SegmentationData, pages_to_keep: list[int] = None) -> "PdfData":
        try:
            file_content: str = open(xml_file.xml_file_path).read()
        except FileNotFoundError:
            return PdfData.get_blank()

        if pages_to_keep:
            xml_file_content = FilterValidSegmentsPages.filter_xml_pages(file_content, pages_to_keep)
        else:
            xml_file_content = file_content

        pdf_features = PdfFeatures.from_poppler_etree_content(xml_file.xml_file_path, xml_file_content)

        if not pdf_features:
            return PdfData.get_blank()

        pdf_data = PdfData(pdf_features)
        pdf_data.set_segments_from_segmentation_data(segmentation_data)
        pdf_data.set_ml_label_from_segmentation_data(segmentation_data)
        pdf_data.clean_text()
        return pdf_data

    @staticmethod
    def from_texts(texts: list[str]):
        pdf_data = PdfData(None)
        pdf_data.pdf_data_segments = PdfDataSegment.from_texts(texts)
        return pdf_data

    def get_text(self):
        return " ".join([segment.text_content for segment in self.pdf_data_segments if segment.text_content])

    def contains_text(self):
        text = "".join([segment.text_content for segment in self.pdf_data_segments if segment.text_content])
        return text != ""
