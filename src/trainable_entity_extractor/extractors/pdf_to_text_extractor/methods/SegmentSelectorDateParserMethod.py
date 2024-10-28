from trainable_entity_extractor.extractors.pdf_to_text_extractor.methods.SegmentSelectorSameInputOutputMethod import (
    SegmentSelectorSameInputOutputMethod,
)
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.DateParserMethod import DateParserMethod


class SegmentSelectorDateParserMethod(SegmentSelectorSameInputOutputMethod):

    SEMANTIC_METHOD = DateParserMethod
