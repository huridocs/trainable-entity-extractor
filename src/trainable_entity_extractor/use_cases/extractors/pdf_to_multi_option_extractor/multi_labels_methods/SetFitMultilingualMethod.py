from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.use_cases.extractors.ExtractorBase import ExtractorBase
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.multi_labels_methods.SetFitEnglishMethod import (
    SetFitEnglishMethod,
)


class SetFitMultilingualMethod(SetFitEnglishMethod):
    model_name = "sentence-transformers/multi-qa-distilbert-dot-v1"

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        if not extraction_data.multi_value:
            return False

        if ExtractorBase.is_multilingual(extraction_data):
            return True

        return False
