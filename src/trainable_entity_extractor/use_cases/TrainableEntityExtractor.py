from time import time

from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.use_cases.extractors.ExtractorBase import ExtractorBase
from trainable_entity_extractor.use_cases.extractors.NaiveExtractor import NaiveExtractor
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
        NaiveExtractor,
    ]

    def __init__(self, extraction_identifier: ExtractionIdentifier):
        self.extraction_identifier = extraction_identifier
        self.multi_value = False
        self.options = list()

    def train(self, extraction_data: ExtractionData) -> (bool, str):
        start = time()
        send_logs(self.extraction_identifier, f"Set data in {round(time() - start, 2)} seconds")

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
            return success, message

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
