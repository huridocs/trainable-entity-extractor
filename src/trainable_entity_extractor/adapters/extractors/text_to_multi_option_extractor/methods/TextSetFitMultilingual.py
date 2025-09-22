from trainable_entity_extractor.domain.ExtractionData import ExtractionData

from trainable_entity_extractor.ports.ExtractorBase import ExtractorBase
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.methods.TextSetFit import TextSetFit


class TextSetFitMultilingual(TextSetFit):
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        if not extraction_data.multi_value:
            return False

        if ExtractorBase.is_multilingual(extraction_data):
            return True

        return False
