from abc import ABC, abstractmethod
from typing import Optional, Any


class JobRunner(ABC):

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.retry_count = 0
        self.job_id: Optional[str] = None

    @abstractmethod
    def start_job(self) -> str:
        pass

    @abstractmethod
    def get_status(self) -> str:
        pass

    @abstractmethod
    def get_result(self) -> Any:
        pass

    @abstractmethod
    def cancel(self) -> None:
        pass

    def handle_retry_if_possible(self) -> bool:
        if self.retry_count < self.max_retries:
            self.retry_count += 1
            self.job_id = self.start_job()
            return True
        return False

    def is_completed(self) -> bool:
        status = self.get_status()
        return status in ["SUCCESS", "FAILURE"]

    def is_successful(self) -> bool:
        return self.get_status() == "SUCCESS"

    def has_failed(self) -> bool:
        return self.get_status() == "FAILURE"
