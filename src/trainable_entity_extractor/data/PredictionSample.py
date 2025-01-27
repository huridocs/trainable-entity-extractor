from dataclasses import dataclass

from trainable_entity_extractor.data.PdfData import PdfData
from trainable_entity_extractor.data.PdfDataSegment import PdfDataSegment


@dataclass
class PredictionSample:
    pdf_data: PdfData = None
    segment_selector_texts: list[str] = None
    source_text: str = ""
    entity_name: str = ""

    def get_text(self):
        texts = list()
        for segment in self.pdf_data.pdf_data_segments:
            texts.append(segment.text_content)

        return " ".join(texts)

    @staticmethod
    def from_pdf_data(pdf_data: PdfData):
        return PredictionSample(pdf_data=pdf_data)

    @staticmethod
    def from_text(text: str, entity_name: str = ""):
        pdf_data = PdfData(None)
        pdf_data.pdf_data_segments.append(PdfDataSegment.from_text(text))
        return PredictionSample(segment_selector_texts=[text], entity_name=entity_name, pdf_data=pdf_data, source_text=text)

    @staticmethod
    def from_texts(texts: list[str]):
        return PredictionSample(segment_selector_texts=texts)
