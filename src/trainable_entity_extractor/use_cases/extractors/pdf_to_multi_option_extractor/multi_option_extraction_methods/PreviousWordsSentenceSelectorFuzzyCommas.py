from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyCommas import (
    FuzzyCommas,
)
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.PreviousWordsSegmentSelector import (
    PreviousWordsSegmentSelector,
)
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.SentenceSelectorFuzzyCommas import (
    SentenceSelectorFuzzyCommas,
)


class PreviousWordsSentenceSelectorFuzzyCommas(SentenceSelectorFuzzyCommas):
    def train(self, multi_option_data: ExtractionData):
        self.set_parameters(multi_option_data)
        extraction_data_by_sentences = self.get_extraction_data_by_sentence(multi_option_data)
        marked_segments = list()
        for sample in extraction_data_by_sentences.samples:
            marked_segments.extend(self.get_marked_segments(sample))
        PreviousWordsSegmentSelector(self.extraction_identifier).create_model(marked_segments)
        FuzzyCommas().train(extraction_data_by_sentences)

    def predict(self, multi_option_data: ExtractionData) -> list[list[Value]]:
        extraction_data_by_sentences = self.get_extraction_data_by_sentence(multi_option_data)
        self.set_parameters(extraction_data_by_sentences)
        self.extraction_data = self.get_prediction_data(extraction_data_by_sentences)
        prediction = FuzzyCommas().predict(self.extraction_data)
        return prediction

    def get_prediction_data(self, extraction_data: ExtractionData) -> ExtractionData:
        segment_selector = PreviousWordsSegmentSelector(self.extraction_identifier)
        predict_samples = list()
        for sample in extraction_data.samples:
            segments = self.fix_two_pages_segments(sample)
            selected_segments = segment_selector.predict(segments)
            self.mark_segments_for_context(selected_segments)

            pdf_data = PdfData(file_name=sample.pdf_data.file_name)
            pdf_data.pdf_data_segments = selected_segments

            training_sample = TrainingSample(pdf_data=pdf_data, labeled_data=sample.labeled_data)
            predict_samples.append(training_sample)

        return ExtractionData(
            samples=predict_samples,
            options=self.extraction_data.options,
            multi_value=self.extraction_data.multi_value,
            extraction_identifier=self.extraction_identifier,
        )
