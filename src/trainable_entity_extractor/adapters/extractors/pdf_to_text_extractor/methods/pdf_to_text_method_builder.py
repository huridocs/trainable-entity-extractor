from trainable_entity_extractor.adapters.extractors.ToTextExtractorMethod import ToTextExtractorMethod
from trainable_entity_extractor.adapters.extractors.segment_selector.SegmentSelectorBase import SegmentSelectorBase
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.DateParserMethod import DateParserMethod
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.DateParserWithBreaksMethod import (
    DateParserWithBreaksMethod,
)

from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.GlinerDateParserMethod import (
    GlinerDateParserMethod,
)
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.NerFirstAppearanceMethod import (
    NerFirstAppearanceMethod,
)
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.NerLastAppearanceMethod import (
    NerLastAppearanceMethod,
)
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.NoSpacesRegexMethod import (
    NoSpacesRegexMethod,
)
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.RegexMethod import RegexMethod
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.RegexSubtractionMethod import (
    RegexSubtractionMethod,
)
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.SameInputOutputMethod import (
    SameInputOutputMethod,
)

text_to_text_methods = [
    SameInputOutputMethod,
    RegexMethod,
    NoSpacesRegexMethod,
    RegexSubtractionMethod,
    GlinerDateParserMethod,
    DateParserWithBreaksMethod,
    DateParserMethod,
    NerFirstAppearanceMethod,
    NerLastAppearanceMethod,
]


def pdf_to_text_method_builder(
    pdf_to_text_method: type[SegmentSelectorBase], text_to_text_method: type[ToTextExtractorMethod]
):
    name = pdf_to_text_method.__name__ + text_to_text_method.__name__
    new_pdf_to_text_method = type(name, (pdf_to_text_method,), {})

    def get_model_folder_name():
        return f"{pdf_to_text_method.__name__}_{text_to_text_method.__name__}"

    setattr(new_pdf_to_text_method, "SEMANTIC_METHOD", text_to_text_method)
    setattr(new_pdf_to_text_method, "get_model_folder_name", get_model_folder_name)

    def gpu_needed(self):
        return True

    if "T5" in text_to_text_method.__name__:
        setattr(new_pdf_to_text_method, "gpu_needed", gpu_needed)

    return new_pdf_to_text_method
