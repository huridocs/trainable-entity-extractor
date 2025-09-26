from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.NerFirstAppearanceMethod import (
    NerFirstAppearanceMethod,
)


class NerLastAppearanceMethod(NerFirstAppearanceMethod):
    @staticmethod
    def get_appearance(prediction_texts):
        return prediction_texts[-1] if prediction_texts else ""
