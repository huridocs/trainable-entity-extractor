from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.extractors.ExtractorBase import ExtractorBase
from trainable_entity_extractor.extractors.text_to_multi_option_extractor.methods.TextSingleLabelSetFit import (
    TextSingleLabelSetFit,
)


class TextSingleLabelSetFitMultilingual(TextSingleLabelSetFit):
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        if extraction_data.multi_value:
            return False

        if ExtractorBase.is_multilingual(extraction_data):
            return True

        return False

    def train(self, extraction_data: ExtractionData):
        if ExtractorBase.is_multilingual(extraction_data):
            super().train(extraction_data)
