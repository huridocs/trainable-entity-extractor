from typing import Optional

from pydantic import BaseModel

from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.PdfDataSegment import PdfDataSegment


class PredictionSample(BaseModel):
    pdf_data: Optional[PdfData] = None
    segment_selector_texts: Optional[list[str]] = None
    source_text: str = ""
    entity_name: str = ""

    def get_segments_text(self):
        texts = list()
        for segment in self.pdf_data.pdf_data_segments:
            texts.append(segment.text_content)

        return " ".join(texts)

    def get_input_text(self) -> str:
        return " ".join(self.get_input_text_by_lines())

    def get_input_text_by_lines(self) -> list[str]:
        if self.source_text:
            return [self.source_text]

        if self.segment_selector_texts:
            return self.segment_selector_texts

        return [""]

    @staticmethod
    def from_pdf_data(pdf_data: PdfData):
        return PredictionSample(pdf_data=pdf_data)

    @staticmethod
    def from_text(text: str, entity_name: str = ""):
        pdf_data = PdfData()
        pdf_data.pdf_data_segments.append(PdfDataSegment.from_text(text))
        return PredictionSample(segment_selector_texts=[text], entity_name=entity_name, pdf_data=pdf_data, source_text=text)

    @staticmethod
    def from_texts(texts: list[str]):
        return PredictionSample(segment_selector_texts=texts)
