from trainable_entity_extractor.config import GEMINI_API_KEY
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import (
    TextToMultiOptionMethod,
)
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.methods.gemini_multi_option.GeminiRunMultiOption import (
    GeminiRunMultiOption,
)
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.Gemini.GeminiSample import GeminiSample
from trainable_entity_extractor.use_cases.send_logs import send_logs


class TextGeminiMultiOption(TextToMultiOptionMethod):
    def can_be_used(self, extraction_data):
        if GEMINI_API_KEY:
            return True
        return False

    def should_be_retrained_with_more_data(self):
        return False

    def train(self, extraction_data: ExtractionData):
        number_of_options = len(self.options)
        options_labels = [option.label for option in self.options]
        gemini_samples = [GeminiSample.from_training_sample(sample, True) for sample in extraction_data.samples]
        gemini_runs = [
            GeminiRunMultiOption(
                mistakes_samples=gemini_samples,
                options=options_labels,
                multi_value=self.multi_value,
                from_class_name=self.method_name,
            )
        ]
        sizes = [number_of_options, min(2 * number_of_options, 15), min(4 * number_of_options, 45)]
        gemini_runs += [
            GeminiRunMultiOption(
                max_training_size=n, options=options_labels, multi_value=self.multi_value, from_class_name=self.method_name
            )
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

    def predict(self, predictions_samples: list[PredictionSample]) -> list[list[Option]]:
        gemini_run = GeminiRunMultiOption.from_extractor_identifier_multioption(
            self.extraction_identifier, self.options, self.multi_value, self.method_name
        )
        gemini_samples = [GeminiSample.from_prediction_sample(sample, True) for sample in predictions_samples]
        predictions = gemini_run.run_code(gemini_samples)
        options_labels_to_option = {option.label: option for option in self.options}
        return [
            [
                options_labels_to_option[option_label]
                for option_label in options_prediction
                if option_label in options_labels_to_option
            ]
            for options_prediction in predictions
        ]
