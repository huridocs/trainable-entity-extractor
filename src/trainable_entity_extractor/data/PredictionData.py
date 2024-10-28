from pydantic import BaseModel

from trainable_entity_extractor.data.SegmentBox import SegmentBox


class PredictionData(BaseModel):
    tenant: str = ""
    id: str = ""
    entity_name: str = ""
    source_text: str = ""
    xml_file_name: str = ""
    page_width: float = 0
    page_height: float = 0
    xml_segments_boxes: list[SegmentBox] = list()

    def to_dict(self):
        prediction_data = self.model_dump()
        prediction_data["xml_segments_boxes"] = [x.to_dict() for x in self.xml_segments_boxes]
        return prediction_data
