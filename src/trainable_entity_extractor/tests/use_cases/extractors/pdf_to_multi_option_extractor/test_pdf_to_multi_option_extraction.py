from unittest import TestCase
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.domain.Value import Value
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.PdfToMultiOptionExtractor import (
    PdfToMultiOptionExtractor,
)


class TestPdfToMultiOptionExtraction(TestCase):
    TENANT = "unit_test"
    extraction_id = "multi_option_extraction_test"

    def test_single_value(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        pdf_data_1 = PdfData.from_texts(["point", "1"])
        pdf_data_2 = PdfData.from_texts(["point 2"])
        pdf_data_3 = PdfData.from_texts(["point 3"])

        samples = [
            TrainingSample(pdf_data=pdf_data_1, labeled_data=LabeledData(values=[options[0]])),
            TrainingSample(pdf_data=pdf_data_2, labeled_data=LabeledData(values=[options[1]])),
            TrainingSample(pdf_data=pdf_data_3, labeled_data=LabeledData(values=[options[2]])),
        ]

        multi_option_data = ExtractionData(
            multi_value=False, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        multi_option_extraction = PdfToMultiOptionExtractor(extraction_identifier)
        multi_option_extraction.create_model(multi_option_data)

        prediction_sample_1 = PredictionSample(pdf_data=pdf_data_1)
        prediction_sample_3 = PredictionSample(pdf_data=pdf_data_3)
        suggestions = multi_option_extraction.get_suggestions([prediction_sample_1, prediction_sample_3])

        self.assertEqual(2, len(suggestions))
        self.assertEqual([Value(id="1", label="1", segment_text='<p class="ix_matching_paragraph"><span class="ix_match">1</span></p>')], suggestions[0].values)
        self.assertEqual([Value(id="3", label="3", segment_text='<p class="ix_matching_paragraph">point <span class="ix_match">3</span></p>')], suggestions[1].values)

    def test_multi_value(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        pdf_data_1 = PdfData.from_texts(["point 1 point 2"])
        pdf_data_2 = PdfData.from_texts(["point 2"])
        pdf_data_3 = PdfData.from_texts(["point 3 point 1"])

        samples = [
            TrainingSample(pdf_data=pdf_data_1, labeled_data=LabeledData(values=[options[0], options[1]])),
            TrainingSample(pdf_data=pdf_data_2, labeled_data=LabeledData(values=[options[1]])),
            TrainingSample(pdf_data=pdf_data_3, labeled_data=LabeledData(values=[options[2], options[0]])),
        ]

        multi_option_data = ExtractionData(
            multi_value=True, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        multi_option_extraction = PdfToMultiOptionExtractor(extraction_identifier)
        multi_option_extraction.create_model(multi_option_data)

        prediction_sample_1 = PredictionSample(pdf_data=pdf_data_1)
        prediction_sample_3 = PredictionSample(pdf_data=pdf_data_3)
        suggestions = multi_option_extraction.get_suggestions([prediction_sample_1, prediction_sample_3])

        self.assertEqual(2, len(suggestions))
        self.assertTrue(Value(id="1", label="1", segment_text='<p class="ix_matching_paragraph">point <span class="ix_match">1</span> point 2</p>') in suggestions[0].values)
        self.assertTrue(Value(id="2", label="2", segment_text='<p class="ix_matching_paragraph">point 1 point <span class="ix_match">2</span></p>') in suggestions[0].values)
        self.assertTrue(Value(id="3", label="3", segment_text='<p class="ix_paragraph">point 1 point 2</p>') not in suggestions[0].values)
        self.assertTrue(Value(id="3", label="3", segment_text='<p class="ix_matching_paragraph">point <span class="ix_match">3</span> point 1</p>') in suggestions[1].values)
        self.assertTrue(Value(id="2", label="2", segment_text='<p class="ix_paragraph">point 3 point 1</p>') not in suggestions[1].values)
        self.assertTrue(Value(id="1", label="1", segment_text='<p class="ix_matching_paragraph">point 3 point <span class="ix_match">1</span></p>') in suggestions[1].values)

    def test_no_prediction_data(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        pdf_data_1 = PdfData.from_texts(["point one point two"])
        pdf_data_2 = PdfData.from_texts(["point two"])
        pdf_data_3 = PdfData.from_texts(["point three point one"])

        samples = [
            TrainingSample(pdf_data=pdf_data_1, labeled_data=LabeledData(values=[options[0]])),
            TrainingSample(pdf_data=pdf_data_2, labeled_data=LabeledData(values=[options[1]])),
            TrainingSample(pdf_data=pdf_data_3, labeled_data=LabeledData(values=[options[2]])),
        ]

        multi_option_data = ExtractionData(
            multi_value=True, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        multi_option_extraction = PdfToMultiOptionExtractor(extraction_identifier)
        multi_option_extraction.create_model(multi_option_data)

        suggestions = multi_option_extraction.get_suggestions([])

        self.assertEqual(0, len(suggestions))
