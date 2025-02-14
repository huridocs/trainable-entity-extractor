from pdf_token_type_labels.TokenType import TokenType
from pydantic import BaseModel

from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.SegmentBox import SegmentBox
from trainable_entity_extractor.domain.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.filter_segments_methods.Beginning750 import (
    Beginning750,
)
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.filter_segments_methods.End750 import (
    End750,
)


class Suggestion(BaseModel):
    tenant: str
    id: str
    xml_file_name: str = ""
    entity_name: str = ""
    text: str = ""
    empty_suggestion: bool = False
    values: list[Option] = list()
    segment_text: str = ""
    page_number: int = 1
    segments_boxes: list[SegmentBox] = list()

    def is_empty(self):
        if self.empty_suggestion:
            return True

        return not self.text and not self.values

    def mark_suggestion_if_empty(self):
        if self.is_empty():
            self.empty_suggestion = True

        return self

    def to_output(self):
        suggestion_dict = self.model_dump()
        suggestion_dict["segments_boxes"] = [x.to_output() for x in self.segments_boxes]
        return suggestion_dict

    @staticmethod
    def get_empty(extraction_identifier: ExtractionIdentifier, entity_name: str) -> "Suggestion":
        return Suggestion(
            tenant=extraction_identifier.run_name,
            id=extraction_identifier.extraction_name,
            xml_file_name=entity_name,
            entity_name=entity_name,
        )

    def add_prediction(self, text: str, prediction_pdf_data: PdfData):
        self.add_segments(prediction_pdf_data)
        self.text = text

    def add_prediction_multi_option(self, training_sample: TrainingSample, values: list[Option], context_from_the_end: bool):
        self.add_segments(training_sample.pdf_data, context_from_the_end)
        self.values = values

    def add_segments(self, pdf_data: PdfData, context_from_the_end: bool = False):
        context_segments: list[PdfDataSegment] = [x for x in pdf_data.pdf_data_segments if x.ml_label]
        valid_types = [TokenType.LIST_ITEM, TokenType.TITLE, TokenType.TEXT, TokenType.SECTION_HEADER, TokenType.CAPTION]
        context_segments = [x for x in context_segments if x.segment_type in valid_types]

        if not context_segments:
            self.page_number = 1
            return

        if context_from_the_end:
            context_segments = End750().filter_segments(context_segments)
        else:
            context_segments = Beginning750().filter_segments(context_segments)

        self.page_number = context_segments[0].page_number
        pages = pdf_data.pdf_features.pages if pdf_data.pdf_features else []
        self.segments_boxes = [SegmentBox.from_pdf_segment(pdf_segment, pages) for pdf_segment in context_segments]
        self.segment_text = " ________ ".join([pdf_segment.text_content for pdf_segment in context_segments])

    def scale_up(self):
        for segment_box in self.segments_boxes:
            segment_box.scale_up()

        return self

    @staticmethod
    def from_prediction_text(extraction_identifier: ExtractionIdentifier, entity_name: str, text: str):
        suggestion = Suggestion.get_empty(extraction_identifier, entity_name)
        suggestion.text = text
        return suggestion

    @staticmethod
    def from_prediction_multi_option(extraction_identifier: ExtractionIdentifier, entity_name: str, values: list[Option]):
        suggestion = Suggestion.get_empty(extraction_identifier, entity_name)
        suggestion.values = values
        return suggestion
