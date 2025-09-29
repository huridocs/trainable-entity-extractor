from enum import StrEnum


class JobType(StrEnum):
    TRAIN = "TRAIN"
    PREDICT = "PREDICT"
    PERFORMANCE = "PERFORMANCE"
