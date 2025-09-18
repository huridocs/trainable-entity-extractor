from enum import StrEnum


class JobStatus(StrEnum):
    """Enumeration for job statuses"""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    CANCELLED = "CANCELLED"
    RETRY = "RETRY"
