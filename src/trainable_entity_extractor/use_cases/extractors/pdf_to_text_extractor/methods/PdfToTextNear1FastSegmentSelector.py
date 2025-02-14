from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.use_cases.extractors.pdf_to_text_extractor.methods.PdfToTextFastSegmentSelector import (
    PdfToTextFastSegmentSelector,
)
from trainable_entity_extractor.use_cases.extractors.segment_selector.Near1FastSegmentSelector import (
    Near1FastSegmentSelector,
)


class PdfToTextNear1FastSegmentSelector(PdfToTextFastSegmentSelector):

    SEGMENT_SELECTOR = Near1FastSegmentSelector

    def create_segment_selector_model(self, extraction_data):
        segments = list()

        for sample in extraction_data.samples:
            segments.extend(sample.pdf_data.pdf_data_segments)

        fast_segment_selector = self.SEGMENT_SELECTOR(self.extraction_identifier)
        fast_segment_selector.create_model(segments=segments)
        return True, ""

    def predict(self, predictions_samples: list[PredictionSample]) -> list[str]:
        if not predictions_samples:
            return [""] * len(predictions_samples)

        fast_segment_selector = self.SEGMENT_SELECTOR(self.extraction_identifier)

        for sample in predictions_samples:
            selected_segments = fast_segment_selector.predict(sample.pdf_data.pdf_data_segments)
            self.mark_predicted_segments(selected_segments)
            sample.segment_selector_texts = self.get_predicted_texts(sample.pdf_data)

        semantic_metadata_extraction = self.SEMANTIC_METHOD(self.extraction_identifier, self.get_name())
        return semantic_metadata_extraction.predict(predictions_samples)
