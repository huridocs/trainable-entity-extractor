from trainable_entity_extractor.use_cases.extractors.pdf_to_text_extractor.methods.PdfToTextNear1FastSegmentSelector import (
    PdfToTextNear1FastSegmentSelector,
)
from trainable_entity_extractor.use_cases.extractors.segment_selector.Near4FastSegmentSelector import (
    Near4FastSegmentSelector,
)


class PdfToTextNear4FastSegmentSelector(PdfToTextNear1FastSegmentSelector):

    SEGMENT_SELECTOR = Near4FastSegmentSelector
