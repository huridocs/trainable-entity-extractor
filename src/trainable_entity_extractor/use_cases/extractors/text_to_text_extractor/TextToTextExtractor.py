from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.use_cases.extractors.ToTextExtractor import ToTextExtractor
from trainable_entity_extractor.use_cases.extractors.ToTextExtractorMethod import ToTextExtractorMethod
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.DateParserMethod import DateParserMethod
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.DateParserWithBreaksMethod import (
    DateParserWithBreaksMethod,
)
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.Gemini.GeminiTextMethod import (
    GeminiTextMethod,
)
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.GlinerDateParserMethod import (
    GlinerDateParserMethod,
)
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.InputWithoutSpaces import (
    InputWithoutSpaces,
)
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.MT5TrueCaseEnglishSpanishMethod import (
    MT5TrueCaseEnglishSpanishMethod,
)
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.NerFirstAppearanceMethod import (
    NerFirstAppearanceMethod,
)
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.NerLastAppearanceMethod import (
    NerLastAppearanceMethod,
)
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.NoSpacesRegexMethod import (
    NoSpacesRegexMethod,
)
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.RegexMethod import RegexMethod
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.RegexSubtractionMethod import (
    RegexSubtractionMethod,
)
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.SameInputOutputMethod import (
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

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        for sample in extraction_data.samples:
            if sample.segment_selector_texts or sample.labeled_data.source_text:
                return True

        return False

    def create_model(self, extraction_data: ExtractionData) -> tuple[bool, str]:
        if not extraction_data or not extraction_data.samples:
            return super().create_model(extraction_data)

        for sample in extraction_data.samples:
            if not sample.segment_selector_texts and sample.labeled_data.source_text:
                sample.segment_selector_texts = [sample.labeled_data.source_text]

        return super().create_model(extraction_data)

    @staticmethod
    def set_segment_selector_texts(predictions_samples):
        for sample in predictions_samples:
            if not sample.segment_selector_texts and sample.source_text:
                sample.segment_selector_texts = [sample.source_text]

    def get_suggestions(self, predictions_samples: list[PredictionSample]) -> list[Suggestion]:
        self.set_segment_selector_texts(predictions_samples)
        return super().get_suggestions(predictions_samples)
