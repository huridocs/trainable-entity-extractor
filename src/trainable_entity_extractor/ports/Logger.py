from abc import ABC, abstractmethod
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LogSeverity import LogSeverity


class Logger(ABC):
    @abstractmethod
    def log(
        self,
        extraction_identifier: ExtractionIdentifier,
        message: str,
        severity: LogSeverity = LogSeverity.info,
        exception: Exception = None,
    ):
        pass
