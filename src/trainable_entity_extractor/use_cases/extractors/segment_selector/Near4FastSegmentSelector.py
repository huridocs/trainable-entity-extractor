from trainable_entity_extractor.use_cases.extractors.segment_selector.Near1FastSegmentSelector import (
    Near1FastSegmentSelector,
)


class Near4FastSegmentSelector(Near1FastSegmentSelector):
    NUMBER_OF_NEIGHBORS = 4
