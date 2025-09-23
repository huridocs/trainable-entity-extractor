from trainable_entity_extractor.adapters.extractors.segment_selector.SegmentSelector import SegmentSelector
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.ports.ExtractorBase import ExtractorBase


class PdfToTextSegmentSelector(SegmentSelector):

    def __init__(self, extraction_identifier: ExtractionIdentifier):
        super().__init__(extraction_identifier)

    def train(self, extraction_data: ExtractionData):
        samples_with_label_segments_boxes = [x for x in extraction_data.samples if x.labeled_data.label_segments_boxes]
        extraction_data_with_samples = ExtractorBase.get_extraction_data_from_samples(
            extraction_data, samples_with_label_segments_boxes
        )
        success, error = self.create_segment_selector_model(extraction_data_with_samples)

        if not success:
            return

        for sample in extraction_data.samples:
            sample.segment_selector_texts = [x.text_content for x in sample.pdf_data.pdf_data_segments if x.ml_label]

        semantic_metadata_extraction = self.SEMANTIC_METHOD(self.extraction_identifier, self.get_name())
        semantic_metadata_extraction.train(extraction_data_with_samples)

    def predict(self, predictions_samples: list[PredictionSample]) -> list[str]:
        segment_selector = SegmentSelector(self.extraction_identifier)
        if not segment_selector.model or not predictions_samples:
            return [""] * len(predictions_samples)

        segment_selector.set_extraction_segments([x.pdf_data for x in predictions_samples])

        for sample in predictions_samples:
            sample.segment_selector_texts = self.get_predicted_texts(sample.pdf_data)

        semantic_metadata_extraction = self.SEMANTIC_METHOD(self.extraction_identifier, self.get_name())
        return semantic_metadata_extraction.predict(predictions_samples)

    def create_segment_selector_model(self, extraction_data: ExtractionData):
        segment_selector = SegmentSelector(self.extraction_identifier)
        pdfs_data = [sample.pdf_data for sample in extraction_data.samples]
        return segment_selector.create_model(pdfs_data=pdfs_data)

    @staticmethod
    def get_predicted_texts(pdf_data: PdfData) -> list[str]:
        predicted_pdf_segments = [x for x in pdf_data.pdf_data_segments if x.ml_label]

        tags_texts: list[str] = list()
        for pdf_segment in predicted_pdf_segments:
            for page, token in pdf_data.pdf_features.loop_tokens():
                if pdf_segment.intersects(PdfDataSegment.from_pdf_token(token)):
                    tags_texts.append(token.content.strip())

        return tags_texts
