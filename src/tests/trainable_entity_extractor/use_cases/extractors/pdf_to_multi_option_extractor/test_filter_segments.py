from unittest import TestCase
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanBeginningDigits3000 import (
    CleanBeginningDigits3000,
)
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanBeginningDot1000 import (
    CleanBeginningDot1000,
)
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanBeginningDot250 import (
    CleanBeginningDot250,
)
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanEndDot1000 import (
    CleanEndDot1000,
)
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanEndDot250 import (
    CleanEndDot250,
)


class TestFilterSegments(TestCase):
    TENANT = "multi_option_extraction_test"
    extraction_id = "extraction_id"

    def get_data(self) -> ExtractionData:
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        pdf_data_1 = PdfData.from_texts(["point 1"])
        pdf_data_2 = PdfData.from_texts(["point 2", "point 2"])
        pdf_data_3 = PdfData.from_texts(["point 3", "point 3", "point 3"])

        samples = [
            TrainingSample(pdf_data=pdf_data_1, labeled_data=LabeledData(values=[options[0]])),
            TrainingSample(pdf_data=pdf_data_2, labeled_data=LabeledData(values=[options[1]])),
            TrainingSample(pdf_data=pdf_data_3, labeled_data=LabeledData(values=[options[2]])),
        ]

        multi_option_data = ExtractionData(
            multi_value=False, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        return multi_option_data

    def get_data_for_context(self) -> ExtractionData:
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        pdf_data = PdfData.from_texts(["point 1"] * 3000)

        samples = [
            TrainingSample(pdf_data=pdf_data, labeled_data=LabeledData(values=[options[0]])),
        ]

        multi_option_data = ExtractionData(
            multi_value=False, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        return multi_option_data

    def get_empty_segments(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        pdf_data_1 = PdfData.from_texts([""])

        samples = [
            TrainingSample(pdf_data=pdf_data_1, labeled_data=LabeledData(values=[options[0]])),
        ]

        multi_option_data = ExtractionData(
            multi_value=False, options=options, samples=samples, extraction_identifier=extraction_identifier
        )
        return multi_option_data

    def get_no_segments(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        pdf_data_1 = PdfData()

        samples = [
            TrainingSample(pdf_data=pdf_data_1, labeled_data=LabeledData(values=[options[0]])),
        ]

        multi_option_data = ExtractionData(
            multi_value=False, options=options, samples=samples, extraction_identifier=extraction_identifier
        )
        return multi_option_data

    def test_filters(self):
        methods = [
            CleanBeginningDigits3000(),
            CleanBeginningDot250(),
            CleanEndDot250(),
            CleanBeginningDot1000(),
            CleanEndDot1000(),
        ]

        for method in methods:
            multi_option_data_filtered = method.filter(self.get_data())
            first_pdf_data_segments = multi_option_data_filtered.samples[0].pdf_data.pdf_data_segments
            second_pdf_data_segments = multi_option_data_filtered.samples[1].pdf_data.pdf_data_segments
            third_pdf_data_segments = multi_option_data_filtered.samples[2].pdf_data.pdf_data_segments

            self.assertEqual(3, len(multi_option_data_filtered.samples))
            self.assertEqual(1, len(first_pdf_data_segments))
            self.assertEqual(2, len(second_pdf_data_segments))
            self.assertEqual(3, len(third_pdf_data_segments))
            self.assertTrue(first_pdf_data_segments[0].text_content in ["point 1", "point"])
            self.assertTrue(second_pdf_data_segments[0].text_content in ["point 2", "point"])
            self.assertTrue(third_pdf_data_segments[0].text_content in ["point 3", "point"])

    def test_empty_text(self):
        methods = [
            CleanBeginningDigits3000(),
            CleanBeginningDot250(),
            CleanEndDot250(),
            CleanBeginningDot1000(),
            CleanEndDot1000(),
        ]

        for method in methods:
            multi_option_data_filtered = method.filter(self.get_empty_segments())
            first_pdf_data_segments = multi_option_data_filtered.samples[0].pdf_data.pdf_data_segments

            self.assertEqual(1, len(multi_option_data_filtered.samples))
            self.assertEqual(1, len(first_pdf_data_segments))
            self.assertEqual("no text", first_pdf_data_segments[0].text_content)

    def test_no_segments(self):
        methods = [
            CleanBeginningDigits3000(),
            CleanBeginningDot250(),
            CleanEndDot250(),
            CleanBeginningDot1000(),
            CleanEndDot1000(),
        ]

        for method in methods:
            multi_option_data_filtered = method.filter(self.get_no_segments())
            first_pdf_data_segments = multi_option_data_filtered.samples[0].pdf_data.pdf_data_segments

            self.assertEqual(1, len(multi_option_data_filtered.samples))
            self.assertEqual(1, len(first_pdf_data_segments))
            self.assertEqual("no text", first_pdf_data_segments[0].text_content)

    def test_context_beginning(self):
        methods = [
            CleanBeginningDigits3000(),
            CleanBeginningDot250(),
            CleanBeginningDot1000(),
        ]

        for method in methods:
            multi_option_data = self.get_data_for_context()
            method.filter(multi_option_data)
            original_data_segments = multi_option_data.samples[0].pdf_data.pdf_data_segments

            for i in range(1, 45):
                self.assertEqual(1, original_data_segments[i].ml_label)

            for i in range(375, 3000):
                self.assertEqual(0, original_data_segments[i].ml_label)

    def test_context_end(self):
        methods = [CleanEndDot250(), CleanEndDot1000()]

        for method in methods:
            multi_option_data = self.get_data_for_context()
            method.filter(multi_option_data)
            original_data_segments = multi_option_data.samples[0].pdf_data.pdf_data_segments

            for i in range(1, 2000):
                self.assertEqual(0, original_data_segments[i].ml_label)

            for i in range(2955, 3000):
                self.assertEqual(1, original_data_segments[i].ml_label)
