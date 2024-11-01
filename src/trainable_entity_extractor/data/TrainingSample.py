from dataclasses import dataclass

from trainable_entity_extractor.data.LabeledData import LabeledData
from trainable_entity_extractor.data.Option import Option
from trainable_entity_extractor.data.PdfData import PdfData


@dataclass
class TrainingSample:
    pdf_data: PdfData = None
    labeled_data: LabeledData = None
    segment_selector_texts: list[str] = None

    def get_text(self):
        texts = list()
        for pdf_metadata_segment in self.pdf_data.pdf_data_segments:
            texts.append(pdf_metadata_segment.text_content)

        return " ".join(texts)

    @staticmethod
    def from_text(source_text: str, label_text: str, language_iso: str = "en"):
        labeled_data = LabeledData(source_text=source_text, label_text=label_text, language_iso=language_iso)
        return TrainingSample(labeled_data=labeled_data)

    @staticmethod
    def from_values(source_text: str, values: list[Option], language_iso: str = "en"):
        labeled_data = LabeledData(source_text=source_text, values=values, language_iso=language_iso)
        return TrainingSample(labeled_data=labeled_data)
