from trainable_entity_extractor.extractors.pdf_to_text_extractor.methods.SegmentSelectorDateParserMethod import (
    SegmentSelectorSameInputOutputMethod,
)
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.DateParserWithBreaksMethod import (
    DateParserWithBreaksMethod,
)


class SegmentSelectorDateParserWithBreaksMethod(SegmentSelectorSameInputOutputMethod):

    SEMANTIC_METHOD = DateParserWithBreaksMethod
