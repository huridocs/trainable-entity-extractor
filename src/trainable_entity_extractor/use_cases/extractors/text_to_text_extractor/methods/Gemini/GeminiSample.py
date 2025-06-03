from pydantic import BaseModel


class GeminiSample(BaseModel):
    input_text: str
    output_text: str = ""
    __hash__ = object.__hash__

    @staticmethod
    def from_prediction_sample(prediction_sample):
        return GeminiSample(input_text=prediction_sample.get_input_text())

    @staticmethod
    def from_training_sample(training_sample):
        return GeminiSample(
            input_text=" ".join(training_sample.get_input_text_by_lines()),
            output_text=training_sample.labeled_data.label_text,
        )
