from trainable_entity_extractor.extractors.pdf_to_text_extractor.methods.FastSegmentSelectorInputWithoutSpaces import (
    FastSegmentSelectorInputWithoutSpaces,
)
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.MT5TrueCaseEnglishSpanishMethod import (
    MT5TrueCaseEnglishSpanishMethod,
)


class FastSegmentSelectorMT5TrueCaseEnglishSpanishMethod(FastSegmentSelectorInputWithoutSpaces):

    SEMANTIC_METHOD = MT5TrueCaseEnglishSpanishMethod
