from trainable_entity_extractor.domain.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.adapters.extractors.segment_selector.FastAndPositionsSegmentSelector import (
    FastAndPositionsSegmentSelector,
)


class Near1FastSegmentSelector(FastAndPositionsSegmentSelector):
    NUMBER_OF_NEIGHBORS = 1

    def predictions_scores_to_segments(self, segments: list[PdfDataSegment], prediction_scores: list[float]):
        predicted_segments = []
        for i, (segment, prediction) in enumerate(zip(segments, prediction_scores)):
            if prediction > 0.5:
                predicted_segments.append(segment)
                continue

            start_index = max(0, i - self.NUMBER_OF_NEIGHBORS)
            end_index = i + 1 + self.NUMBER_OF_NEIGHBORS
            neighbor_scores = prediction_scores[start_index:i] + prediction_scores[i + 1 : end_index]

            if any(score > 0.5 for score in neighbor_scores):
                predicted_segments.append(segment)

        return predicted_segments
