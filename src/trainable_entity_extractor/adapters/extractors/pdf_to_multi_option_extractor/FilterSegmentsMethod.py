from abc import ABC, abstractmethod

from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample


class FilterSegmentsMethod(ABC):
    def get_name(self):
        return self.__class__.__name__

    @abstractmethod
    def filter_segments(self, pdf_data_segments: list[PdfDataSegment]) -> list[PdfDataSegment]:
        pass

    def filter(self, multi_option_data: ExtractionData) -> ExtractionData:
        filtered_samples: list[TrainingSample] = list()
        for sample in multi_option_data.samples:
            filtered_pdf_data = PdfData(
                pdf_features=sample.pdf_data.pdf_features,
                file_name=sample.pdf_data.file_name,
                file_type=sample.pdf_data.file_type,
            )

            filtered_pdf_data.pdf_data_segments = self.filter_segments(sample.pdf_data.pdf_data_segments)

            filtered_samples.append(TrainingSample(pdf_data=filtered_pdf_data, labeled_data=sample.labeled_data))

        return ExtractionData(
            samples=filtered_samples, options=multi_option_data.options, multi_value=multi_option_data.multi_value
        )

    def filter_prediction_samples(self, prediction_samples: PredictionSamplesData) -> PredictionSamplesData:
        filtered_samples: list[PredictionSample] = list()
        for sample in prediction_samples.prediction_samples:
            filtered_pdf_data = PdfData(
                pdf_features=sample.pdf_data.pdf_features,
                file_name=sample.pdf_data.file_name,
                file_type=sample.pdf_data.file_type,
            )

            filtered_pdf_data.pdf_data_segments = self.filter_segments(sample.pdf_data.pdf_data_segments)

            filtered_samples.append(
                PredictionSample(
                    pdf_data=filtered_pdf_data,
                    entity_name=sample.entity_name,
                    segment_selector_texts=sample.segment_selector_texts,
                    source_text=sample.source_text,
                )
            )

        return PredictionSamplesData(
            prediction_samples=filtered_samples,
            options=prediction_samples.options,
            multi_value=prediction_samples.multi_value,
        )
