from time import time

from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.LogsMessage import Severity
from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.data.Suggestion import Suggestion
from trainable_entity_extractor.extractors.ExtractorBase import ExtractorBase
from trainable_entity_extractor.extractors.NaiveExtractor import NaiveExtractor
from trainable_entity_extractor.extractors.pdf_to_text_extractor.PdfToTextExtractor import PdfToTextExtractor
from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.PdfToMultiOptionExtractor import (
    PdfToMultiOptionExtractor,
)
from trainable_entity_extractor.extractors.text_to_multi_option_extractor.TextToMultiOptionExtractor import (
    TextToMultiOptionExtractor,
)
from trainable_entity_extractor.extractors.text_to_text_extractor.TextToTextExtractor import TextToTextExtractor
from trainable_entity_extractor.send_logs import send_logs


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
            return extractor_instance.create_model(extraction_data)

        send_logs(self.extraction_identifier, "Error creating extractor", Severity.error)

        return False, "Error creating extractor"

    def predict(self, prediction_samples: list[PredictionSample]) -> list[Suggestion]:
        extractor_name = self.extraction_identifier.get_extractor_used()
        if not extractor_name:
            send_logs(self.extraction_identifier, f"No extractor available", Severity.error)
            return []

        for extractor in self.EXTRACTORS:
            extractor_instance = extractor(self.extraction_identifier)
            if extractor_instance.get_name() != extractor_name:
                continue

            suggestions = extractor_instance.get_suggestions(prediction_samples)
            suggestions = [suggestion.mark_suggestion_if_empty() for suggestion in suggestions]
            message = f"Using {extractor_instance.get_name()} to calculate {len(suggestions)} suggestions"
            send_logs(self.extraction_identifier, message)
            return suggestions

        send_logs(self.extraction_identifier, f"No extractor available", Severity.error)
        return []
