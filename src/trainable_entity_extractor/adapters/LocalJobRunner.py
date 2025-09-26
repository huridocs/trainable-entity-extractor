import uuid
import time
from typing import Optional, Any
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.ports.JobRunner import JobRunner


class LocalJobRunner(JobRunner):
    def __init__(self, extractor_job: TrainableEntityExtractorJob, max_retries: int = 3):
        super().__init__(max_retries)
        self.extractor_job = extractor_job
        self.result = None
        self.status = "pending"
        self.start_time = None
        self.end_time = None

    def start_job(self) -> str:
        """Start the job and return a job ID"""
        self.job_id = f"job_{uuid.uuid4().hex[:8]}"
        self.status = "running"
        self.start_time = time.time()
        return self.job_id

    def get_status(self) -> str:
        """Get the current status of the job"""
        return self.status

    def get_result(self) -> Any:
        """Get the result of the job"""
        return self.result

    def cancel(self) -> None:
        """Cancel the job"""
        self.status = "cancelled"
        self.end_time = time.time()

    def complete_job(self, result: Any, success: bool = True):
        """Mark the job as completed with a result"""
        self.result = result
        self.status = "completed" if success else "failed"
        self.end_time = time.time()

    def get_execution_time(self) -> int:
        """Get the execution time in seconds"""
        if self.start_time and self.end_time:
            return int(self.end_time - self.start_time)
        elif self.start_time:
            return int(time.time() - self.start_time)
        return 0
