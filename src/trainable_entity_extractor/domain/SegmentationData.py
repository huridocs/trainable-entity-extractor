from pydantic import BaseModel

from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionData import PredictionData
from trainable_entity_extractor.domain.SegmentBox import SegmentBox


class SegmentationData(BaseModel):
    page_width: float
    page_height: float
    xml_segments_boxes: list[SegmentBox]
    label_segments_boxes: list[SegmentBox]

    @staticmethod
    def from_labeled_data(labeled_data: LabeledData) -> "SegmentationData":
        return SegmentationData(
            page_width=labeled_data.page_width,
            page_height=labeled_data.page_height,
            xml_segments_boxes=labeled_data.xml_segments_boxes,
            label_segments_boxes=labeled_data.label_segments_boxes,
        )

    @staticmethod
    def from_prediction_data(prediction_data: PredictionData) -> "SegmentationData":
        return SegmentationData(
            page_width=prediction_data.page_width,
            page_height=prediction_data.page_height,
            xml_segments_boxes=prediction_data.xml_segments_boxes,
            label_segments_boxes=[],
        )

    def rescale(self, page_width, page_height):
        factor_width = page_width / self.page_width
        factor_height = page_height / self.page_height
        self.page_width = page_width
        self.page_height = page_height
        self.xml_segments_boxes = [
            xml_segment_box.rescaled(factor_width, factor_height) for xml_segment_box in self.xml_segments_boxes
        ]
        self.label_segments_boxes = [
            xml_segment_box.rescaled(factor_width, factor_height) for xml_segment_box in self.label_segments_boxes
        ]
