from trainable_entity_extractor.extractors.pdf_to_text_extractor.methods.FastSegmentSelectorInputWithoutSpaces import (
    FastSegmentSelectorInputWithoutSpaces,
)
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.NerFirstAppearanceMethod import (
    NerFirstAppearanceMethod,
)


class FastSegmentSelectorNerFirstAppearanceMethod(FastSegmentSelectorInputWithoutSpaces):

    SEMANTIC_METHOD = NerFirstAppearanceMethod
