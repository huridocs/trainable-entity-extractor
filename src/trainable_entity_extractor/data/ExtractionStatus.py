from enum import Enum


class ExtractionStatus(Enum):
    NO_MODEL = 0
    PROCESSING = 1
    READY = 2
