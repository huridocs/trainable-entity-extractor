from typing import Optional

from trainable_entity_extractor.domain.DistributedSubJob import DistributedSubJob


class JobSelectorUseCase:
    @staticmethod
    def select_best_job(distributed_job: DistributedJob) -> Optional[DistributedSubJob]:
        successful_evaluations = [sub_job for sub_job in distributed_job.sub_jobs if sub_job.status == JobStatus.SUCCESS]

        if not successful_evaluations:
            return None

        best_job: DistributedSubJob | None = None
        highest_score = -1.0

        for sub_job in successful_evaluations:
            if sub_job.result and hasattr(sub_job.result, "performance"):
                if sub_job.result.is_perfect:
                    return sub_job

                score = sub_job.result.performance
                if score > highest_score:
                    highest_score = score
                    best_job = sub_job

        return best_job
