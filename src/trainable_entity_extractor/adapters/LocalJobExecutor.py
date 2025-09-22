from typing import Tuple, List

from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.PdfToMultiOptionExtractor import (
    PdfToMultiOptionExtractor,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_text_extractor.PdfToTextExtractor import PdfToTextExtractor
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.TextToMultiOptionExtractor import (
    TextToMultiOptionExtractor,
)
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.TextToTextExtractor import TextToTextExtractor
from trainable_entity_extractor.domain.DistributedJob import DistributedJob
from trainable_entity_extractor.domain.DistributedSubJob import DistributedSubJob
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.domain.JobStatus import JobStatus
from trainable_entity_extractor.ports.JobExecutor import JobExecutor
from trainable_entity_extractor.ports.ExtractorBase import ExtractorBase
from trainable_entity_extractor.use_cases.TrainUseCase import TrainUseCase
from trainable_entity_extractor.use_cases.PredictUseCase import PredictUseCase


class LocalJobExecutor(JobExecutor):
    EXTRACTORS: list[type[ExtractorBase]] = [
        PdfToMultiOptionExtractor,
        TextToMultiOptionExtractor,
        PdfToTextExtractor,
        TextToTextExtractor,
    ]

    def start_performance_evaluation(
        self, extraction_identifier: ExtractionIdentifier, distributed_sub_job: DistributedSubJob
    ):
        try:
            extraction_data = self.data_retriever.get_extraction_data(extraction_identifier)
            if not extraction_data:
                distributed_sub_job.status = JobStatus.FAILURE
                return None

            train_use_case = TrainUseCase(extractors=self.EXTRACTORS)
            performance = train_use_case.get_performance(distributed_sub_job.extractor_job, extraction_data)
            if performance:
                distributed_sub_job.result = performance
                distributed_sub_job.status = JobStatus.SUCCESS
                return performance
        except Exception as e:
            distributed_sub_job.status = JobStatus.FAILURE
            return None

        distributed_sub_job.status = JobStatus.FAILURE
        return None

    def start_training(
        self, extraction_identifier: ExtractionIdentifier, distributed_sub_job: DistributedSubJob
    ) -> Tuple[bool, str]:
        try:
            extraction_data = self.data_retriever.get_extraction_data(extraction_identifier)
            if not extraction_data:
                distributed_sub_job.status = JobStatus.FAILURE
                return False, "No extraction data available for training"

            train_use_case = TrainUseCase(extractors=self.EXTRACTORS)
            success, message = train_use_case.train_one_method(distributed_sub_job.extractor_job, extraction_data)
            if success:
                distributed_sub_job.status = JobStatus.SUCCESS
                return True, "Training completed successfully"
        except Exception as e:
            distributed_sub_job.status = JobStatus.FAILURE
            return False, str(e)

        distributed_sub_job.status = JobStatus.FAILURE
        return False, "Training failed"

    def start_prediction(self, extraction_identifier: ExtractionIdentifier, distributed_sub_job: DistributedSubJob) -> None:
        try:
            prediction_data = self.data_retriever.get_prediction_data(extraction_identifier)
            if not prediction_data:
                distributed_sub_job.status = JobStatus.FAILURE
                distributed_sub_job.result = False
                return

            predict_use_case = PredictUseCase(extractors=self.EXTRACTORS)
            suggestions = predict_use_case.predict(distributed_sub_job.extractor_job, prediction_data)
            success = self.data_retriever.save_suggestions(extraction_identifier, suggestions)
            if success:
                distributed_sub_job.status = JobStatus.SUCCESS
                distributed_sub_job.result = True
            else:
                distributed_sub_job.status = JobStatus.FAILURE
                distributed_sub_job.result = False
        except Exception as e:
            distributed_sub_job.status = JobStatus.FAILURE
            distributed_sub_job.result = False

    def execute_prediction_with_samples(
        self, extractor_job: TrainableEntityExtractorJob, prediction_samples: List[PredictionSample]
    ) -> Tuple[bool, str, List[Suggestion]]:
        job_id = f"predict_samples_{extractor_job.method_name}"
        extractor_job.status = JobStatus.RUNNING

        try:
            extractor_instance = self._get_extractor_instance(extractor_job.extractor_name)
            if not extractor_instance:
                extractor_job.status = JobStatus.FAILURE
                return False, f"Extractor {extractor_job.extractor_name} not found", []

            suggestions = extractor_instance.get_suggestions(prediction_samples)
            suggestions = [suggestion.mark_suggestion_if_empty() for suggestion in suggestions]

            extractor_job.status = JobStatus.SUCCESS
            self.prediction_results[job_id] = suggestions
            return True, "Prediction with samples completed successfully", suggestions

        except Exception as e:
            extractor_job.status = JobStatus.FAILURE
            return False, str(e), []

    def update_job_statuses(self, job: DistributedJob):
        pass

    def cancel_jobs(self, job: DistributedJob) -> None:
        for sub_job in job.sub_jobs:
            if sub_job.status not in self.get_finished_status():
                sub_job.status = JobStatus.CANCELED
