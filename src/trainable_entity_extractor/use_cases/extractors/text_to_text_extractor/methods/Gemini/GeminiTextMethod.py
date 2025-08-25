from trainable_entity_extractor.config import GEMINI_API_KEY
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.use_cases.extractors.ToTextExtractorMethod import ToTextExtractorMethod
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.Gemini.GeminiRun import GeminiRun
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.Gemini.GeminiSample import GeminiSample


class GeminiTextMethod(ToTextExtractorMethod):

    def should_be_retrained_with_more_data(self):
        return False

    def train(self, extraction_data: ExtractionData):
        if not GEMINI_API_KEY:
            return

        gemini_samples = [GeminiSample.from_training_sample(sample) for sample in extraction_data.samples]
        gemini_runs = [GeminiRun(mistakes_samples=gemini_samples, from_class_name=self.from_class_name)]
        gemini_runs += [GeminiRun(max_training_size=n, from_class_name=self.from_class_name) for n in [5, 15, 45]]

        for previous_gemini_run, gemini_run in zip(gemini_runs, gemini_runs[1:]):
            gemini_run.run_training(previous_gemini_run)

            if not gemini_run.mistakes_samples:
                break

        gemini_with_code = [run for run in gemini_runs if run.code]

        if not gemini_with_code:
            return

        gemini_with_code.sort(key=lambda run: len(run.mistakes_samples))
        gemini_with_code[0].save_code(self.extraction_identifier)

    def predict(self, predictions_samples: list[PredictionSample]) -> list[str]:
        gemini_run = GeminiRun.from_extractor_identifier(self.extraction_identifier, self.from_class_name)
        gemini_samples = [GeminiSample.from_prediction_sample(sample) for sample in predictions_samples]
        return gemini_run.run_code(gemini_samples)
