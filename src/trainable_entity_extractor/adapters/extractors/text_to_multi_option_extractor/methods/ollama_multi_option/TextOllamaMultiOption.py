from trainable_entity_extractor.config import OLLAMA_API_KEY
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import (
    TextToMultiOptionMethod,
)
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.methods.ollama_multi_option.OllamaRunMultiOption import (
    OllamaRunMultiOption,
)
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.Ollama.OllamaSample import (
    OllamaSample,
)


class TextOllamaMultiOption(TextToMultiOptionMethod):
    def get_model_folder_name(self):
        return "TextOllamaMultiOption"

    def can_be_used(self, extraction_data):
        if OLLAMA_API_KEY:
            return True
        return False

    def should_be_retrained_with_more_data(self):
        return False

    def train(self, extraction_data: ExtractionData):
        number_of_options = len(extraction_data.options)
        options_labels = [option.label for option in extraction_data.options]
        ollama_samples = [OllamaSample.from_training_sample(sample, True) for sample in extraction_data.samples]
        ollama_runs = [
            OllamaRunMultiOption(
                mistakes_samples=ollama_samples, options=options_labels, multi_value=extraction_data.multi_value
            )
        ]
        sizes = [number_of_options, min(2 * number_of_options, 15), min(4 * number_of_options, 45)]
        ollama_runs += [
            OllamaRunMultiOption(max_training_size=n, options=options_labels, multi_value=extraction_data.multi_value)
            for n in sizes
        ]

        for previous_ollama_run, ollama_run in zip(ollama_runs, ollama_runs[1:]):
            ollama_run.run_training(previous_ollama_run)
            if not ollama_run.mistakes_samples:
                break

        ollama_with_code = [run for run in ollama_runs if run.code]

        if not ollama_with_code:
            return

        ollama_with_code.sort(key=lambda run: len(run.mistakes_samples))
        ollama_with_code[0].save_code(self.extraction_identifier)

    def predict(self, prediction_samples: PredictionSamplesData) -> list[list[Option]]:
        self.options = prediction_samples.options
        self.multi_value = prediction_samples.multi_value
        ollama_run = OllamaRunMultiOption.from_extractor_identifier_multioption(
            self.extraction_identifier, prediction_samples.options, prediction_samples.multi_value
        )
        ollama_samples = [
            OllamaSample.from_prediction_sample(sample, True) for sample in prediction_samples.prediction_samples
        ]
        predictions = ollama_run.run_code(ollama_samples)
        options_labels_to_option = {option.label: option for option in prediction_samples.options}
        return [
            [
                options_labels_to_option[option_label]
                for option_label in options_prediction
                if option_label in options_labels_to_option
            ]
            for options_prediction in predictions
        ]
