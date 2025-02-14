from trainable_entity_extractor.domain.PdfDataSegment import PdfDataSegment

from trainable_entity_extractor.use_cases.extractors.segment_selector.FastSegmentSelector import FastSegmentSelector


class FastAndPositionsSegmentSelector(FastSegmentSelector):
    def get_features(self, segment: PdfDataSegment, segments: list[PdfDataSegment]):
        features = list()
        text = segment.text_content

        if segment in self.text_segments:
            index = self.text_segments.index(segment)
            previous_segment_texts = self.clean_texts(self.text_segments[index - 1]) if index > 0 else []
            next_segment_texts = (
                self.clean_texts(self.text_segments[index + 1]) if index + 1 < len(self.text_segments) else []
            )
        else:
            index = segments.index(segment)
            previous_segment_texts = self.clean_texts(segments[index - 1]) if index > 0 else ""
            next_segment_texts = self.clean_texts(segments[index + 1]) if index + 1 < len(segments) else ""

        for word in self.previous_words:
            features.append(1 if word in previous_segment_texts else 0)

        for word in self.next_words:
            features.append(1 if word in next_segment_texts else 0)

        commas_percentage = len([x for x in text if x == ","]) / len(text) if text else 0
        features.append(commas_percentage)
        features.append(segment.page_number)
        features.append(segment.bounding_box.top)
        features.append(segment.bounding_box.left)
        features.append(segment.bounding_box.width)
        features.append(segment.bounding_box.height)
        features.append(segment.bounding_box.right)
        features.append(segment.bounding_box.bottom)

        return features
