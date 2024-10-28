from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.extractors.ExtractorBase import ExtractorBase
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_labels_methods.SetFitEnglishMethod import (
    SetFitEnglishMethod,
)


class SetFitMultilingualMethod(SetFitEnglishMethod):
    model_name = "Alibaba-NLP/gte-multilingual-reranker-base"

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        if not extraction_data.multi_value:
            return False

        if ExtractorBase.is_multilingual(extraction_data):
            return True

        return False
