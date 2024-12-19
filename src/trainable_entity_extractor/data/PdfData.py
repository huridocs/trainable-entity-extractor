from typing import Optional

from pdf_features.PdfToken import PdfToken
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType

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
        segments_tokens: dict[PdfDataSegment, list[PdfToken]] = dict()
        segmentation_regions: list[PdfDataSegment] = [
            segment_box.to_pdf_segment() for segment_box in segmentation_data.xml_segments_boxes
        ]
        for page, token in self.pdf_features.loop_tokens():
            segment_from_token: PdfDataSegment = PdfDataSegment.from_pdf_token(token)
            intersects_segmentation = [region for region in segmentation_regions if region.intersects(segment_from_token)]

            if not intersects_segmentation:
                segments_tokens[PdfDataSegment.from_pdf_token(token)] = [token]
                continue

            segment_from_token.segment_type = intersects_segmentation[0].segment_type
            segments_tokens.setdefault(intersects_segmentation[0], []).append(token)

        segments_tokens = {segment: self.remove_super_scripts(tokens) for segment, tokens in segments_tokens.items()}
        segments_tokens = {
            segment: self.merge_tokens_that_belongs_to_same_word(tokens) for segment, tokens in segments_tokens.items()
        }
        segments = [PdfDataSegment.from_token_list_to_merge(tokens) for tokens in segments_tokens.values()]
        self.pdf_data_segments.extend(segments)
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
        return "" != self.get_text()

    @staticmethod
    def remove_super_scripts(segment_tokens: list[PdfToken]) -> list[PdfToken]:
        if not segment_tokens:
            return []

        font_sizes = [token.font.font_size for token in segment_tokens]

        if PdfData.similar_font_sizes(font_sizes):
            return segment_tokens

        tokens_no_super_scripts = []

        min_left = min([token.bounding_box.left for token in segment_tokens])

        for i, token in enumerate(segment_tokens):
            if token.bounding_box.left == min_left:
                tokens_no_super_scripts.append(token)
                continue

            if token.token_type in [
                TokenType.FORMULA,
                TokenType.FOOTNOTE,
                TokenType.TABLE,
                TokenType.PICTURE,
                TokenType.PAGE_FOOTER,
            ]:
                tokens_no_super_scripts.append(token)
                continue

            if token.font.font_size == min(font_sizes) and token.content.isnumeric() and float(token.content) < 999:
                continue

            tokens_no_super_scripts.append(token)

        return tokens_no_super_scripts

    @staticmethod
    def similar_font_sizes(fonts_sizes: list[float]) -> bool:
        return max(fonts_sizes) - min(fonts_sizes) < 1.5

    @staticmethod
    def merge_tokens_that_belongs_to_same_word(segment_tokens: list[PdfToken]) -> list[PdfToken]:
        if not segment_tokens:
            return []

        merged_tokens: list[PdfToken] = [segment_tokens[0]]

        for token in segment_tokens[1:]:
            if token.page_number != merged_tokens[-1].page_number:
                merged_tokens.append(token)
                continue

            if merged_tokens[-1].bounding_box.get_vertical_intersection(token.bounding_box) < 4:
                merged_tokens.append(token)
                continue

            if 0 < merged_tokens[-1].bounding_box.get_horizontal_distance(token.bounding_box):
                merged_tokens.append(token)
                continue

            merged_tokens[-1].content += token.content
            merged_tokens[-1].bounding_box = Rectangle.merge_rectangles([merged_tokens[-1].bounding_box, token.bounding_box])

        return merged_tokens
