from pydantic import BaseModel

from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.TrainingSample import TrainingSample


class GeminiSample(BaseModel):
    input_text: str
    output: str | list[str] = ""
    __hash__ = object.__hash__

    @staticmethod
    def from_prediction_sample(prediction_sample: PredictionSample) -> "GeminiSample":
        return GeminiSample(input_text=prediction_sample.get_input_text())

    @staticmethod
    def from_training_sample(training_sample: TrainingSample, multioption: bool = False) -> "GeminiSample":
        if multioption:
            return GeminiSample(
                input_text=" ".join(training_sample.get_input_text_by_lines()),
                output=(
                    [value.label for value in training_sample.labeled_data.values]
                    if training_sample.labeled_data.values
                    else []
                ),
            )

        return GeminiSample(
            input_text=" ".join(training_sample.get_input_text_by_lines()),
            output=training_sample.labeled_data.label_text if training_sample.labeled_data.label_text else "",
        )
