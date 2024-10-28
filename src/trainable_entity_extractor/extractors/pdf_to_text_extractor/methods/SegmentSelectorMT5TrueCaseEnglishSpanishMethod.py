from trainable_entity_extractor.extractors.pdf_to_text_extractor.methods.SegmentSelectorDateParserMethod import (
    SegmentSelectorSameInputOutputMethod,
)
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.MT5TrueCaseEnglishSpanishMethod import (
    MT5TrueCaseEnglishSpanishMethod,
)


class SegmentSelectorMT5TrueCaseEnglishSpanishMethod(SegmentSelectorSameInputOutputMethod):

    SEMANTIC_METHOD = MT5TrueCaseEnglishSpanishMethod
