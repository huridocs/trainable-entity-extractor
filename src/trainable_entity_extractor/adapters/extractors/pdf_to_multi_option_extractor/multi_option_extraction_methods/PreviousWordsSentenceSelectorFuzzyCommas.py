from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyCommas import (
    FuzzyCommas,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.PreviousWordsSegmentSelector import (
    PreviousWordsSegmentSelector,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.SentenceSelectorFuzzyCommas import (
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
        FuzzyCommas(self.extraction_identifier).train(extraction_data_by_sentences)

    def predict(self, prediction_samples_data: PredictionSamplesData) -> list[list[Value]]:
        temp_samples = []
        for sample in prediction_samples_data.prediction_samples:
            from trainable_entity_extractor.domain.LabeledData import LabeledData

            labeled_data = LabeledData(values=[], xml_file_name=sample.entity_name, id="temp")
            temp_sample = TrainingSample(pdf_data=sample.pdf_data, labeled_data=labeled_data)
            temp_samples.append(temp_sample)

        temp_extraction_data = ExtractionData(
            samples=temp_samples,
            extraction_identifier=self.extraction_identifier,
            options=prediction_samples_data.options,
            multi_value=prediction_samples_data.multi_value,
        )

        extraction_data_by_sentences = self.get_extraction_data_by_sentence(temp_extraction_data)
        self.set_parameters(extraction_data_by_sentences)
        self.prediction_samples_data = self.get_prediction_samples_data(
            extraction_data_by_sentences, prediction_samples_data.options, prediction_samples_data.multi_value
        )
        prediction = FuzzyCommas(self.extraction_identifier).predict(self.prediction_samples_data)
        return prediction

    def get_prediction_samples_data(self, extraction_data: ExtractionData, options, multi_value) -> PredictionSamplesData:
        segment_selector = PreviousWordsSegmentSelector(self.extraction_identifier)
        predict_samples = list()
        for sample in extraction_data.samples:
            segments = self.fix_two_pages_segments(sample)
            selected_segments = segment_selector.predict(segments)
            self.mark_segments_for_context(selected_segments)

            pdf_data = PdfData(file_name=sample.pdf_data.file_name)
            pdf_data.pdf_data_segments = selected_segments

            prediction_sample = PredictionSample(
                pdf_data=pdf_data,
                entity_name=sample.labeled_data.xml_file_name if sample.labeled_data else "unknown",
            )
            predict_samples.append(prediction_sample)

        return PredictionSamplesData(
            prediction_samples=predict_samples,
            options=options,
            multi_value=multi_value,
        )
