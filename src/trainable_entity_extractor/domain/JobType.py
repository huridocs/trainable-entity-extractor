from enum import StrEnum


class JobType(StrEnum):
    """Enumeration for job types"""

    TRAIN = "TRAIN"
    PREDICT = "PREDICT"
    PERFORMANCE = "PERFORMANCE"
