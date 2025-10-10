import json
from dateparser.search import search_dates
from gliner import GLiNER


class GlinerDateExtractor:

    def __init__(self, model):
        self.model = model

    @staticmethod
    def find_unique_entity_dicts(entities: list[dict]) -> list[dict]:
        dicts_without_score = [{k: v for k, v in d.items() if k != "score"} for d in entities]
        return list({json.dumps(d, sort_keys=True): d for d in dicts_without_score}.values())

    @staticmethod
    def remove_overlapping_entities(entities):
        sorted_entities = sorted(entities, key=lambda x: (x["start"], -len(x["text"])))

        result = []
        last_end = -1

        for entity in sorted_entities:
            if entity["start"] >= last_end:
                result.append(entity)
                last_end = entity["end"]

        return result

    def extract_dates(self, text: str):
        words = text.split()

        entities = []
        window_size = 50
        slide_size = 25
        last_slide_end_index = 0

        for i in range(0, len(words), slide_size):
            window_words = words[i : i + window_size]
            window_text = " ".join(window_words)
            window_entities = self.model.predict_entities(window_text, ["date"])

            for entity in window_entities:
                entity["start"] += last_slide_end_index
                entity["end"] += last_slide_end_index

            slide_words = words[i : i + slide_size]
            slide_text = " ".join(slide_words)
            last_slide_end_index += len(slide_text) + 1
            entities.extend(window_entities)

        entities = self.find_unique_entity_dicts(entities)
        entities = [e for e in entities if search_dates(e["text"])]
        entities = self.remove_overlapping_entities(entities)
        date_times = [d[1] for e in entities for d in search_dates(e["text"])]
        return date_times

    @staticmethod
    def get_model():
        return GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
