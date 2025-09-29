from enum import StrEnum


class JobStatus(StrEnum):
    WAITING = "WAITING"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    CANCELED = "CANCELED"
    RETRY = "RETRY"
