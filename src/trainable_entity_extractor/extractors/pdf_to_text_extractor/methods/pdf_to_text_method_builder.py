from trainable_entity_extractor.extractors.ToTextExtractorMethod import ToTextExtractorMethod
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.DateParserMethod import DateParserMethod
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.DateParserWithBreaksMethod import (
    DateParserWithBreaksMethod,
)
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.GlinerDateParserMethod import (
    GlinerDateParserMethod,
)
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.RegexMethod import RegexMethod
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.RegexSubtractionMethod import (
    RegexSubtractionMethod,
)
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.SameInputOutputMethod import SameInputOutputMethod

text_to_text_methods = [
    SameInputOutputMethod,
    RegexMethod,
    RegexSubtractionMethod,
    GlinerDateParserMethod,
    DateParserWithBreaksMethod,
    DateParserMethod,
    # Error downloading the model
    # NerFirstAppearanceMethod,
    # NerLastAppearanceMethod,
]


def pdf_to_text_method_builder(
    pdf_to_text_method: type[ToTextExtractorMethod], text_to_text_method: type[ToTextExtractorMethod]
):
    name = pdf_to_text_method.__name__ + text_to_text_method.__name__
    new_pdf_to_text_method = type(name, (pdf_to_text_method,), {})
    setattr(new_pdf_to_text_method, "SEMANTIC_METHOD", text_to_text_method)
    return new_pdf_to_text_method
