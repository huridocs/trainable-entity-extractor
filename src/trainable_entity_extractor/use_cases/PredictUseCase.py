from trainable_entity_extractor.config import DATA_PATH
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.ports.ExtractorBase import ExtractorBase
from trainable_entity_extractor.ports.Logger import Logger


class PredictUseCase:
    def __init__(self, extractors: list[type[ExtractorBase]], logger: Logger):
        self.extractors: list[type[ExtractorBase]] = extractors
        self.logger = logger

    def predict(self, extractor_job: TrainableEntityExtractorJob, samples: list[PredictionSample]) -> list[Suggestion]:
        extraction_identifier = ExtractionIdentifier(
            run_name=extractor_job.run_name, output_path=DATA_PATH, extraction_name=extractor_job.extraction_name
        )

        extractor_name = extractor_job.extractor_name
        for extractor in self.extractors:
            extractor_instance = extractor(extraction_identifier, self.logger)
            if extractor_instance.get_name() != extractor_name:
                continue

            prediction_samples = PredictionSamplesData(
                prediction_samples=samples, options=extractor_job.options, multi_value=extractor_job.multi_value
            )
            return extractor_instance.get_suggestions(extractor_job.method_name, prediction_samples)

        return []
