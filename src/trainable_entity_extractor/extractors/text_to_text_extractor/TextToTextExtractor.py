from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.extractors.ToTextExtractor import ToTextExtractor
from trainable_entity_extractor.extractors.ToTextExtractorMethod import ToTextExtractorMethod
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.DateParserMethod import DateParserMethod
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.DateParserWithBreaksMethod import (
    DateParserWithBreaksMethod,
)
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.GlinerDateParserMethod import (
    GlinerDateParserMethod,
)
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.InputWithoutSpaces import InputWithoutSpaces
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.MT5TrueCaseEnglishSpanishMethod import (
    MT5TrueCaseEnglishSpanishMethod,
)
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.NerFirstAppearanceMethod import (
    NerFirstAppearanceMethod,
)
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.NerLastAppearanceMethod import (
    NerLastAppearanceMethod,
)
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.RegexMethod import RegexMethod
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.RegexSubtractionMethod import (
    RegexSubtractionMethod,
)
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.SameInputOutputMethod import SameInputOutputMethod


class TextToTextExtractor(ToTextExtractor):
    METHODS: list[type[ToTextExtractorMethod]] = [
        SameInputOutputMethod,
        InputWithoutSpaces,
        RegexMethod,
        RegexSubtractionMethod,
        DateParserWithBreaksMethod,
        DateParserMethod,
        GlinerDateParserMethod,
        NerFirstAppearanceMethod,
        NerLastAppearanceMethod,
        MT5TrueCaseEnglishSpanishMethod,
    ]

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        for sample in extraction_data.samples:
            if sample.tags_texts:
                return True

        return False
