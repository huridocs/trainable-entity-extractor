from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.adapters.extractors.ToTextExtractorMethod import ToTextExtractorMethod
from trainable_entity_extractor.adapters.extractors.pdf_to_text_extractor.methods.PdfToTextSegmentSelector import (
    PdfToTextSegmentSelector,
)

from trainable_entity_extractor.adapters.extractors.segment_selector.FastAndPositionsSegmentSelector import (
    FastAndPositionsSegmentSelector,
)


class PdfToTextFastSegmentSelector(PdfToTextSegmentSelector):

    SEMANTIC_METHOD: type[ToTextExtractorMethod] = None

    def create_segment_selector_model(self, extraction_data):
        segments = list()

        for sample in extraction_data.samples:
            segments.extend(sample.pdf_data.pdf_data_segments)

        fast_segment_selector = FastAndPositionsSegmentSelector(self.extraction_identifier)
        fast_segment_selector.create_model(segments=segments)
        return True, ""

    def predict(self, prediction_samples_data: PredictionSamplesData) -> list[str]:
        predictions_samples = prediction_samples_data.prediction_samples
        if not predictions_samples:
            return [""] * len(predictions_samples)

        self._select_segments([x.pdf_data for x in predictions_samples])

        for sample in predictions_samples:
            sample.segment_selector_texts = self.get_predicted_texts(sample.pdf_data)

        semantic_metadata_extraction = self.SEMANTIC_METHOD(self.extraction_identifier)
        return semantic_metadata_extraction.predict(prediction_samples_data)

    def _select_segments(self, pdfs_data: list[PdfData]):
        if not pdfs_data:
            return

        fast_segment_selector = FastAndPositionsSegmentSelector(self.extraction_identifier)

        for pdf_data in pdfs_data:
            selected_segments = fast_segment_selector.predict(pdf_data.pdf_data_segments)
            self.mark_predicted_segments(selected_segments)

    @staticmethod
    def mark_predicted_segments(segments: list[PdfDataSegment]):
        for segment in segments:
            segment.ml_label = 1
