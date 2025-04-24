from trainable_entity_extractor.domain.PdfDataSegment import PdfDataSegment

from trainable_entity_extractor.use_cases.extractors.segment_selector.FastSegmentSelector import FastSegmentSelector
from rapidfuzz import fuzz


class PreviousWordsSegmentSelector(FastSegmentSelector):
    def create_model(self, segments: list[PdfDataSegment]):
        self.text_segments = [x for x in segments if x.segment_type in self.text_types]
        self.save_predictive_common_words(self.text_segments)

    def predict(self, segments):
        self.text_segments = [x for x in segments if x.segment_type in self.text_types]
        self.load_repeated_words()

        predicted_segments = []

        for i, segment in enumerate(self.text_segments):
            if i > 0:
                previous_segment_texts = self.clean_texts(self.text_segments[i - 1])
                previous_segment_text = " ".join(previous_segment_texts)
            else:
                previous_segment_text = ""

            for word in self.previous_words:
                if fuzz.partial_ratio(word, previous_segment_text) >= 90:
                    predicted_segments.append(segment)
                    break

        return predicted_segments
