from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.data.Suggestion import Suggestion
from trainable_entity_extractor.data.TrainingSample import TrainingSample
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

    def get_suggestions(self, predictions_samples: list[PredictionSample]) -> list[Suggestion]:
        for sample in predictions_samples:
            if not sample.segment_selector_texts and sample.source_text:
                sample.segment_selector_texts = [sample.source_text]

        return super().get_suggestions(predictions_samples)