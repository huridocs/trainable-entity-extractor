from trainable_entity_extractor.config import OLLAMA_API_KEY
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.adapters.extractors.ToTextExtractorMethod import ToTextExtractorMethod
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.Ollama.OllamaRun import OllamaRun
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.Ollama.OllamaSample import OllamaSample


class OllamaTextMethod(ToTextExtractorMethod):

    def get_model_folder_name(self):
        return "OllamaTextMethod"

    def should_be_retrained_with_more_data(self):
        return False

    def train(self, extraction_data: ExtractionData):
        if not OLLAMA_API_KEY:
            return

        ollama_samples = [OllamaSample.from_training_sample(sample) for sample in extraction_data.samples]
        ollama_runs = [OllamaRun(mistakes_samples=ollama_samples)]
        ollama_runs += [OllamaRun(max_training_size=n) for n in [5, 15, 45]]

        for previous_ollama_run, ollama_run in zip(ollama_runs, ollama_runs[1:]):
            ollama_run.run_training(previous_ollama_run)

            if not ollama_run.mistakes_samples:
                break

        ollama_with_code = [run for run in ollama_runs if run.code]

        if not ollama_with_code:
            return

        ollama_with_code.sort(key=lambda run: len(run.mistakes_samples))
        ollama_with_code[0].save_code(self.extraction_identifier)

    def predict(self, prediction_samples_data: PredictionSamplesData) -> list[str]:
        predictions_samples = prediction_samples_data.prediction_samples
        ollama_run = OllamaRun.from_extractor_identifier(self.extraction_identifier)
        ollama_samples = [OllamaSample.from_prediction_sample(sample) for sample in predictions_samples]
        return ollama_run.run_code(ollama_samples)
