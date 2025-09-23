from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FastSegmentSelectorFuzzy95 import (
    FastSegmentSelectorFuzzy95,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyAll75 import (
    FuzzyAll75,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.NextWordsSegmentSelector import (
    NextWordsSegmentSelector,
)


class NextWordsTokenSelectorFuzzy75(FastSegmentSelectorFuzzy95):
    threshold = 75

    def predict(self, prediction_samples_data: PredictionSamplesData) -> list[list[Value]]:
        self.options = prediction_samples_data.options
        self.multi_value = prediction_samples_data.multi_value
        self.get_token_prediction_data(prediction_samples_data)
        segment_selector = NextWordsSegmentSelector(self.extraction_identifier)

        for sample in self.prediction_samples_data.prediction_samples:
            sample.pdf_data.pdf_data_segments = segment_selector.predict(sample.pdf_data.pdf_data_segments)
            self.mark_segments_for_context(sample.pdf_data.pdf_data_segments)

        return FuzzyAll75().predict(self.prediction_samples_data)

    def train(self, multi_option_data: ExtractionData):
        self.set_parameters(multi_option_data)
        self.get_token_extraction_data(multi_option_data)
        marked_segments = list()
        for sample in self.extraction_data.samples:
            marked_segments.extend(self.get_marked_segments(sample))

        NextWordsSegmentSelector(self.extraction_identifier).create_model(marked_segments)

    def get_token_extraction_data(self, extraction_data: ExtractionData):
        samples = list()
        for sample in extraction_data.samples:
            token_segments = []
            if sample.pdf_data.pdf_features and sample.pdf_data.pdf_features.pages:
                for page in sample.pdf_data.pdf_features.pages:
                    token_segments.extend([PdfDataSegment.from_pdf_token(token) for token in page.tokens])

            pdf_data = PdfData(file_name=sample.pdf_data.file_name)
            pdf_data.pdf_data_segments = token_segments

            training_sample = TrainingSample(pdf_data=pdf_data, labeled_data=sample.labeled_data)
            samples.append(training_sample)

        self.extraction_data = ExtractionData(
            samples=samples,
            options=self.options,
            multi_value=self.multi_value,
            extraction_identifier=self.extraction_identifier,
        )

    def get_token_prediction_data(self, prediction_samples_data: PredictionSamplesData):
        samples = list()
        for sample in prediction_samples_data.prediction_samples:
            token_segments = []
            if sample.pdf_data.pdf_features and sample.pdf_data.pdf_features.pages:
                for page in sample.pdf_data.pdf_features.pages:
                    token_segments.extend([PdfDataSegment.from_pdf_token(token) for token in page.tokens])

            pdf_data = PdfData(file_name=sample.pdf_data.file_name)
            pdf_data.pdf_data_segments = token_segments

            prediction_sample = PredictionSample(
                pdf_data=pdf_data,
                entity_name=sample.entity_name,
                segment_selector_texts=sample.segment_selector_texts,
                source_text=sample.source_text,
            )
            samples.append(prediction_sample)

        self.prediction_samples_data = PredictionSamplesData(
            prediction_samples=samples,
            options=prediction_samples_data.options,
            multi_value=prediction_samples_data.multi_value,
        )
