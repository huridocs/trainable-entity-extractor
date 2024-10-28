from trainable_entity_extractor.extractors.text_to_text_extractor.methods.NerFirstAppearanceMethod import (
    NerFirstAppearanceMethod,
)


class NerLastAppearanceMethod(NerFirstAppearanceMethod):
    def get_appearance(self, prediction_texts):
        return prediction_texts[-1] if prediction_texts else ""
