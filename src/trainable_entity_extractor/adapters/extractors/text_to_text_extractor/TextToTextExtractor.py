from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.adapters.extractors.ToTextExtractor import ToTextExtractor
from trainable_entity_extractor.adapters.extractors.ToTextExtractorMethod import ToTextExtractorMethod
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.DateParserMethod import DateParserMethod
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.DateParserWithBreaksMethod import (
    DateParserWithBreaksMethod,
)
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.Gemini.GeminiTextMethod import (
    GeminiTextMethod,
)
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.GlinerDateParserMethod import (
    GlinerDateParserMethod,
)
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.InputWithoutSpaces import (
    InputWithoutSpaces,
)
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.MT5TrueCaseEnglishSpanishMethod import (
    MT5TrueCaseEnglishSpanishMethod,
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


class TextToTextExtractor(ToTextExtractor):
    METHODS: list[type[ToTextExtractorMethod]] = [
        SameInputOutputMethod,
        InputWithoutSpaces,
        RegexMethod,
        NoSpacesRegexMethod,
        RegexSubtractionMethod,
        DateParserWithBreaksMethod,
        DateParserMethod,
        GlinerDateParserMethod,
        NerFirstAppearanceMethod,
        NerLastAppearanceMethod,
        GeminiTextMethod,
        MT5TrueCaseEnglishSpanishMethod,
    ]

    def prepare_for_training(self, extraction_data: ExtractionData) -> tuple[ExtractionData, ExtractionData]:
        """Prepare the extractor for performance evaluation by setting up segment selector texts"""
        # Set up segment selector texts like in create_model
        for sample in extraction_data.samples:
            if not sample.segment_selector_texts and sample.labeled_data.source_text:
                sample.segment_selector_texts = [sample.labeled_data.source_text]
        return self.get_train_test_sets(extraction_data)

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        for sample in extraction_data.samples:
            if sample.segment_selector_texts or sample.labeled_data.label_text:
                return True

        return False
