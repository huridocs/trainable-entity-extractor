from trainable_entity_extractor.data.PdfDataSegment import PdfDataSegment

from trainable_entity_extractor.extractors.segment_selector.FastSegmentSelector import FastSegmentSelector
from rapidfuzz import fuzz


class PreviousWordsSegmentSelector(FastSegmentSelector):
    def create_model(self, segments: list[PdfDataSegment]):
        self.text_segments = [x for x in segments if x.segment_type in self.text_types]
        self.save_predictive_common_words(self.text_segments)

    def predict(self, segments):
        self.text_segments = [x for x in segments if x.paragraph_type in self.text_types]
        self.load_repeated_words()

        predicted_segments = []
        for segment in self.text_segments:

            index = self.text_segments.index(segment)
            previous_segment_texts = self.clean_texts(self.text_segments[index - 1]) if index > 0 else []

            for word in self.previous_words:
                if fuzz.partial_ratio(word, " ".join(previous_segment_texts)) >= 90:
                    predicted_segments.append(segment)
                    break

        return predicted_segments
