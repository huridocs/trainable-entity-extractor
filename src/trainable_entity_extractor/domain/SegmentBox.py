import json

from pdf_features.PdfPage import PdfPage
from pydantic import BaseModel
from pdf_token_type_labels.TokenType import TokenType
from pdf_features.Rectangle import Rectangle

from trainable_entity_extractor.domain.PdfDataSegment import PdfDataSegment

SCALE_RATIO = 0.75


class SegmentBox(BaseModel):
    left: float
    top: float
    width: float
    height: float
    page_number: int
    page_width: int = 0
    page_height: int = 0
    segment_type: TokenType = TokenType.TEXT

    def to_output(self):
        segment_box_dict = json.loads(self.model_dump_json())
        del segment_box_dict["page_width"]
        del segment_box_dict["page_height"]
        del segment_box_dict["segment_type"]
        return segment_box_dict

    def get_bounding_box(self) -> Rectangle:
        return Rectangle.from_width_height(
            left=int(self.left), top=int(self.top), width=int(self.width), height=int(self.height)
        )

    def scale_down(self):
        self.left = round(self.left * SCALE_RATIO, 0)
        self.top = round(self.top * SCALE_RATIO, 0)
        self.width = round(self.width * SCALE_RATIO, 0)
        self.height = round(self.height * SCALE_RATIO, 0)

    def scale_up(self):
        self.left = round(self.left / SCALE_RATIO, 0)
        self.top = round(self.top / SCALE_RATIO, 0)
        self.width = round(self.width / SCALE_RATIO, 0)
        self.height = round(self.height / SCALE_RATIO, 0)

    def to_pdf_segment(self):
        return PdfDataSegment(
            page_number=self.page_number,
            bounding_box=self.get_bounding_box(),
            text_content="",
            segment_type=self.segment_type,
        )

    @staticmethod
    def from_pdf_segment(pdf_segment: PdfDataSegment, pages: list[PdfPage]):
        page_width = pages[0].page_width if pages else 0
        page_height = pages[0].page_height if pages else 0

        if pages and pdf_segment.page_number < len(pages):
            page_width = pages[pdf_segment.page_number].page_width
            page_height = pages[pdf_segment.page_number].page_height

        return SegmentBox(
            left=pdf_segment.bounding_box.left,
            top=pdf_segment.bounding_box.top,
            width=pdf_segment.bounding_box.width,
            height=pdf_segment.bounding_box.height,
            page_number=pdf_segment.page_number,
            page_width=page_width,
            page_height=page_height,
        )
