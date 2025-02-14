from pydantic import BaseModel

from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.SegmentBox import SegmentBox


class LabeledData(BaseModel):
    tenant: str = ""
    id: str = ""
    xml_file_name: str = ""
    entity_name: str = ""
    language_iso: str = ""
    label_text: str = ""
    empty_value: bool = False
    values: list[Option] = list()
    source_text: str = ""
    page_width: float = 0
    page_height: float = 0
    xml_segments_boxes: list[SegmentBox] = list()
    label_segments_boxes: list[SegmentBox] = list()

    def scale_down_labels(self):
        for label in self.label_segments_boxes:
            label.scale_down()

        return self
