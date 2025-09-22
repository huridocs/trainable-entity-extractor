from trainable_entity_extractor.config import DATA_PATH
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.ports.ExtractorBase import ExtractorBase


class PredictUseCase:
    def __init__(self, extractors: list[type[ExtractorBase]]):
        self.extractors: list[type[ExtractorBase]] = extractors

    def predict(self, extractor_job: TrainableEntityExtractorJob, samples: list[PredictionSample]) -> list[Suggestion]:
        extraction_identifier = ExtractionIdentifier(
            run_name=extractor_job.run_name, output_path=DATA_PATH, extraction_name=extractor_job.extraction_name
        )

        extractor_name = extractor_job.extractor_name
        for extractor in self.extractors:
            extractor_instance = extractor(extraction_identifier)
            if extractor_instance.get_name() != extractor_name:
                continue

            return extractor_instance.get_suggestions(samples)

        return []
