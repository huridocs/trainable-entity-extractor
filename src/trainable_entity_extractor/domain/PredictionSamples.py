from pydantic import BaseModel

from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PredictionSample import PredictionSample


class PredictionSamples(BaseModel):
    prediction_samples: list[PredictionSample] = []
    options: list[Option] = []
    multi_value: bool = False
