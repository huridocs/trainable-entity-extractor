from transformers import AutoTokenizer
from trainable_entity_extractor.extractors.text_to_multi_option_extractor.methods.TextBert import TextBert

MODEL_NAME = "google-bert/bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class TextBertLarge(TextBert):

    model_name = "google-bert/bert-large-uncased"
