from pathlib import Path
from typing import Optional

from pdf_features.PdfFeatures import PdfFeatures
from pydantic import BaseModel

from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.SegmentationData import SegmentationData


class TrainingSample(BaseModel):
    pdf_data: Optional[PdfData] = None
    labeled_data: Optional[LabeledData] = None
    segment_selector_texts: Optional[list[str]] = None
    __hash__ = object.__hash__

    def get_segments_text(self):
        texts = list()
        for pdf_metadata_segment in self.pdf_data.pdf_data_segments:
            texts.append(pdf_metadata_segment.text_content)

        return " ".join(texts)

    def get_input_text(self) -> str:
        if self.labeled_data and self.labeled_data.source_text:
            return self.labeled_data.source_text

        if self.segment_selector_texts:
            return " ".join(self.segment_selector_texts)

        return ""

    def get_input_text_by_lines(self) -> list[str]:
        if self.labeled_data and self.labeled_data.source_text:
            return [self.labeled_data.source_text]

        if self.segment_selector_texts:
            return self.segment_selector_texts

        return [""]

    @staticmethod
    def from_text(source_text: str, label_text: str, language_iso: str = "en"):
        labeled_data = LabeledData(source_text=source_text, label_text=label_text, language_iso=language_iso)
        return TrainingSample(labeled_data=labeled_data)

    @staticmethod
    def from_values(source_text: str, values: list[Option], language_iso: str = "en"):
        labeled_data = LabeledData(source_text=source_text, values=values, language_iso=language_iso)
        return TrainingSample(labeled_data=labeled_data)

    @staticmethod
    def from_pdf(pdf_path: str | Path, label_text: str, language_iso: str = "en"):
        pdf_features = PdfFeatures.from_pdf_path(pdf_path)
        pdf_data = PdfData(pdf_features=pdf_features)
        segmentation_data = SegmentationData(
            page_width=pdf_features.pages[0].page_width if pdf_features.pages else 0,
            page_height=pdf_features.pages[0].page_height if pdf_features.pages else 0,
            xml_segments_boxes=[],
            label_segments_boxes=[],
        )
        pdf_data.set_segments_from_segmentation_data(segmentation_data=segmentation_data)
        labeled_data = LabeledData(label_text=label_text, language_iso=language_iso)
        return TrainingSample(pdf_data=pdf_data, labeled_data=labeled_data)
