from trainable_entity_extractor.data.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.extractors.segment_selector.FastAndPositionsSegmentSelector import (
    FastAndPositionsSegmentSelector,
)


class NearFastSegmentSelector(FastAndPositionsSegmentSelector):
    @staticmethod
    def predictions_scores_to_segments(segments: list[PdfDataSegment], prediction_scores: list[float]):
        predicted_segments = []
        for i, (segment, prediction) in enumerate(zip(segments, prediction_scores)):
            if prediction > 0.5:
                predicted_segments.append(segment)
                continue

            if len(prediction_scores) >= i + 1 and prediction_scores[i + 1] > 0.5:
                predicted_segments.append(segment)
                continue

            if i != 0 and prediction_scores[i - 1] > 0.5:
                predicted_segments.append(segment)

        return predicted_segments
