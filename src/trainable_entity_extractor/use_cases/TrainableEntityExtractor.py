import shutil

from trainable_entity_extractor.domain.Performance import Performance
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.use_cases.extractors.ExtractorBase import ExtractorBase
from trainable_entity_extractor.use_cases.extractors.pdf_to_text_extractor.PdfToTextExtractor import PdfToTextExtractor
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.PdfToMultiOptionExtractor import (
    PdfToMultiOptionExtractor,
)
from trainable_entity_extractor.use_cases.extractors.text_to_multi_option_extractor.TextToMultiOptionExtractor import (
    TextToMultiOptionExtractor,
)
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.TextToTextExtractor import TextToTextExtractor
from trainable_entity_extractor.use_cases.send_logs import send_logs


class TrainableEntityExtractor:
    EXTRACTORS: list[type[ExtractorBase]] = [
        PdfToMultiOptionExtractor,
        TextToMultiOptionExtractor,
        PdfToTextExtractor,
        TextToTextExtractor,
    ]

    def __init__(self, extraction_identifier: ExtractionIdentifier):
        self.extraction_identifier = extraction_identifier
        self.multi_value = False
        self.options = list()

    def train(self, extraction_data: ExtractionData) -> (bool, str):
        if extraction_data.extraction_identifier.is_training_canceled():
            send_logs(self.extraction_identifier, "Training canceled", LogSeverity.error)
            return False, "Training canceled"

        if not extraction_data or not extraction_data.samples:
            return False, "No data to create model"

        for extractor in self.EXTRACTORS:
            extractor_instance = extractor(self.extraction_identifier)

            if not extractor_instance.can_be_used(extraction_data):
                continue

            send_logs(self.extraction_identifier, f"Using extractor {extractor_instance.get_name()}")
            send_logs(self.extraction_identifier, f"Creating models with {len(extraction_data.samples)} samples")
            self.extraction_identifier.save_extractor_used(extractor_instance.get_name())
            success, message = extractor_instance.create_model(extraction_data)
            self.extraction_identifier.clean_extractor_folder()
            return success, message

        shutil.rmtree(self.extraction_identifier.get_path(), ignore_errors=True)
        send_logs(self.extraction_identifier, "Error creating extractor", LogSeverity.error)
        return False, "Error creating extractor"

    def predict(self, prediction_samples: list[PredictionSample]) -> list[Suggestion]:
        extractor_name = self.extraction_identifier.get_extractor_used()
        if not extractor_name:
            send_logs(self.extraction_identifier, f"No extractor available", LogSeverity.error)
            return []

        for extractor in self.EXTRACTORS:
            extractor_instance = extractor(self.extraction_identifier)
            if extractor_instance.get_name() != extractor_name:
                continue

            message = f"Using {extractor_instance.get_name()} to calculate {len(prediction_samples)} suggestions"
            send_logs(self.extraction_identifier, message)

            suggestions = extractor_instance.get_suggestions(prediction_samples)
            suggestions = [suggestion.mark_suggestion_if_empty() for suggestion in suggestions]
            return suggestions

        send_logs(self.extraction_identifier, f"No extractor available", LogSeverity.error)
        return []

    def get_distributed_jobs(self, extraction_data: ExtractionData) -> list[TrainableEntityExtractorJob]:
        jobs = list()
        for extractor in self.EXTRACTORS:
            extractor_instance = extractor(self.extraction_identifier)

            if not extractor_instance.can_be_used(extraction_data):
                continue

            send_logs(self.extraction_identifier, f"Getting jobs for extractor {extractor_instance.get_name()}")
            jobs = extractor_instance.get_distributed_jobs(extraction_data)
            break

        return jobs

    def get_performance(self, extractor_job: TrainableEntityExtractorJob, extraction_data: ExtractionData) -> Performance:
        extractor_name = extractor_job.extractor_name
        for extractor in self.EXTRACTORS:
            extractor_instance = extractor(self.extraction_identifier)
            if extractor_instance.get_name() != extractor_name:
                continue

            return extractor_instance.get_performance(extractor_job, extraction_data)

        return Performance()

    def train_one_method(
        self, extractor_job: TrainableEntityExtractorJob, extraction_data: ExtractionData
    ) -> tuple[bool, str]:
        extractor_name = extractor_job.extractor_name
        for extractor in self.EXTRACTORS:
            extractor_instance = extractor(self.extraction_identifier)
            if extractor_instance.get_name() != extractor_name:
                continue

            return extractor_instance.train_one_method(extractor_job, extraction_data)

        return False, f"Extractor {extractor_name} not found"
