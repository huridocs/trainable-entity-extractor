from dataclasses import dataclass

from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.Option import Option
from trainable_entity_extractor.data.TrainingSample import TrainingSample


@dataclass
class ExtractionData:
    samples: list[TrainingSample]
    options: list[Option] = None
    multi_value: bool = False
    extraction_identifier: ExtractionIdentifier = None
