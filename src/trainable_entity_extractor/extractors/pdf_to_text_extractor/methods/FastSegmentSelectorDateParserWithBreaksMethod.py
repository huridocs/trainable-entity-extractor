from trainable_entity_extractor.extractors.pdf_to_text_extractor.methods.FastSegmentSelectorInputWithoutSpaces import (
    FastSegmentSelectorInputWithoutSpaces,
)
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.DateParserWithBreaksMethod import (
    DateParserWithBreaksMethod,
)


class FastSegmentSelectorDateParserWithBreaksMethod(FastSegmentSelectorInputWithoutSpaces):

    SEMANTIC_METHOD = DateParserWithBreaksMethod
