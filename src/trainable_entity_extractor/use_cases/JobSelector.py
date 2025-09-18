from typing import List, Optional

from trainable_entity_extractor.domain.JobPerformance import JobPerformance
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob


class JobSelector:
    @staticmethod
    def select_best_job(
        completed_jobs: List[JobPerformance], total_jobs_count: int, perfect_job: Optional[JobPerformance] = None
    ) -> Optional[TrainableEntityExtractorJob]:
        if perfect_job:
            return perfect_job.extractor_job

        if len(completed_jobs) == total_jobs_count:
            return JobSelector._find_best_performance_job(completed_jobs)

        return None

    @staticmethod
    def _find_best_performance_job(completed_jobs: List[JobPerformance]) -> Optional[TrainableEntityExtractorJob]:
        if not completed_jobs:
            return None

        best_job = max(completed_jobs, key=lambda job: job.performance_score)
        return best_job.extractor_job

    @staticmethod
    def should_cancel_other_jobs(perfect_job: JobPerformance) -> bool:
        return perfect_job.is_perfect
