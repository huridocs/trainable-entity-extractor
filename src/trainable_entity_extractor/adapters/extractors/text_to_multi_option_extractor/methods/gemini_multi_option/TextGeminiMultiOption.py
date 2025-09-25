from trainable_entity_extractor.config import GEMINI_API_KEY
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import (
    TextToMultiOptionMethod,
)
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.methods.gemini_multi_option.GeminiRunMultiOption import (
    GeminiRunMultiOption,
)
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.Gemini.GeminiSample import GeminiSample


class TextGeminiMultiOption(TextToMultiOptionMethod):
    def get_model_folder_name(self):
        return "TextGeminiMultiOption"

    def can_be_used(self, extraction_data):
        if GEMINI_API_KEY:
            return True
        return False

    def should_be_retrained_with_more_data(self):
        return False

    def train(self, extraction_data: ExtractionData):
        number_of_options = len(extraction_data.options)
        options_labels = [option.label for option in extraction_data.options]
        gemini_samples = [GeminiSample.from_training_sample(sample, True) for sample in extraction_data.samples]
        gemini_runs = [
            GeminiRunMultiOption(
                mistakes_samples=gemini_samples, options=options_labels, multi_value=extraction_data.multi_value
            )
        ]
        sizes = [number_of_options, min(2 * number_of_options, 15), min(4 * number_of_options, 45)]
        gemini_runs += [
            GeminiRunMultiOption(max_training_size=n, options=options_labels, multi_value=extraction_data.multi_value)
            for n in sizes
        ]

        for previous_gemini_run, gemini_run in zip(gemini_runs, gemini_runs[1:]):
            gemini_run.run_training(previous_gemini_run)
            if not gemini_run.mistakes_samples:
                break

        gemini_with_code = [run for run in gemini_runs if run.code]

        if not gemini_with_code:
            return

        gemini_with_code.sort(key=lambda run: len(run.mistakes_samples))
        gemini_with_code[0].save_code(self.extraction_identifier)

    def predict(self, prediction_samples: PredictionSamplesData) -> list[list[Option]]:
        self.options = prediction_samples.options
        self.multi_value = prediction_samples.multi_value
        gemini_run = GeminiRunMultiOption.from_extractor_identifier_multioption(
            self.extraction_identifier, prediction_samples.options, prediction_samples.multi_value
        )
        gemini_samples = [
            GeminiSample.from_prediction_sample(sample, True) for sample in prediction_samples.prediction_samples
        ]
        predictions = gemini_run.run_code(gemini_samples)
        options_labels_to_option = {option.label: option for option in prediction_samples.options}
        return [
            [
                options_labels_to_option[option_label]
                for option_label in options_prediction
                if option_label in options_labels_to_option
            ]
            for options_prediction in predictions
        ]
