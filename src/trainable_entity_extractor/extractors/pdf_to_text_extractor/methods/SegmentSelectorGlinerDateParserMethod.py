from trainable_entity_extractor.extractors.pdf_to_text_extractor.methods.SegmentSelectorDateParserMethod import (
    SegmentSelectorSameInputOutputMethod,
)
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.GlinerDateParserMethod import (
    GlinerDateParserMethod,
)


class SegmentSelectorGlinerDateParserMethod(SegmentSelectorSameInputOutputMethod):

    SEMANTIC_METHOD = GlinerDateParserMethod
