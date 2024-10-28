from trainable_entity_extractor.data.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.PreviousWordsSegmentSelector import (
    PreviousWordsSegmentSelector,
)

from rapidfuzz import fuzz


class NextWordsSegmentSelector(PreviousWordsSegmentSelector):
    def predict(self, segments: list[PdfDataSegment]) -> list[PdfDataSegment]:
        self.text_segments = [x for x in segments if x.segment_type in self.text_types]
        self.load_repeated_words()

        predicted_segments = []
        for segment in self.text_segments:

            index = self.text_segments.index(segment)

            next_segment_texts = []
            if index < len(self.text_segments) - 1:
                next_segment_texts = self.clean_texts(self.text_segments[index + 1])

            for word in self.next_words:
                if fuzz.partial_ratio(word, " ".join(next_segment_texts)) >= 90:
                    predicted_segments.append(segment)
                    break

        return predicted_segments
