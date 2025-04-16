from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.SegmentBox import SegmentBox
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.extractors.pdf_to_text_extractor.PdfToTextExtractor import PdfToTextExtractor


class TestPdfToTextExtractor(TestCase):
    TENANT = "TestPdfToTextExtractor"
    extraction_id = "extraction_id"
    extraction_identifier = ExtractionIdentifier(run_name=TENANT, extraction_name=extraction_id)

    def test_no_prediction_data(self):
        pdf_to_text_extractor = PdfToTextExtractor(self.extraction_identifier)
        predictions = pdf_to_text_extractor.get_suggestions([])

        self.assertEqual(0, len(predictions))

    @staticmethod
    def get_samples(count_with_segments: int, count_without_segments: int):
        labeled_data_segments = LabeledData(
            label_text="text", label_segments_boxes=[SegmentBox(left=0, top=0, width=1, height=1, page_number=1)]
        )
        labeled_data_no_segments = LabeledData(label_text="text")
        samples = [TrainingSample(labeled_data=labeled_data_segments)] * count_with_segments
        samples += [TrainingSample(labeled_data=labeled_data_no_segments)] * count_without_segments

        return samples

    def test_get_train_test_with_few_samples(self):
        pdf_to_text_extractor = PdfToTextExtractor(self.extraction_identifier)

        extraction_data = ExtractionData(
            samples=self.get_samples(count_with_segments=4, count_without_segments=5),
            extraction_identifier=self.extraction_identifier,
        )

        train_set, test_set = pdf_to_text_extractor.get_train_test_sets(extraction_data)

        self.assertEqual(train_set.extraction_identifier, self.extraction_identifier)
        self.assertEqual(test_set.extraction_identifier, self.extraction_identifier)
        self.assertEqual(9, len(train_set.samples))
        self.assertEqual(9, len(test_set.samples))

    def test_get_train_test_without_enough_labeled_segments(self):
        pdf_to_text_extractor = PdfToTextExtractor(self.extraction_identifier)

        extraction_data = ExtractionData(
            samples=self.get_samples(count_with_segments=9, count_without_segments=11),
            extraction_identifier=self.extraction_identifier,
        )

        train_set, test_set = pdf_to_text_extractor.get_train_test_sets(extraction_data)

        self.assertEqual(9, len(train_set.samples))
        self.assertEqual(20, len(test_set.samples))

    def test_get_train_test_without_labeled_segments(self):
        pdf_to_text_extractor = PdfToTextExtractor(self.extraction_identifier)

        extraction_data = ExtractionData(
            samples=self.get_samples(count_with_segments=0, count_without_segments=100),
            extraction_identifier=self.extraction_identifier,
        )

        train_set, test_set = pdf_to_text_extractor.get_train_test_sets(extraction_data)

        self.assertEqual(80, len(train_set.samples))
        self.assertEqual(20, len(test_set.samples))

    def test_get_train_test_without_enough_data(self):
        pdf_to_text_extractor = PdfToTextExtractor(self.extraction_identifier)

        extraction_data = ExtractionData(
            samples=self.get_samples(count_with_segments=100, count_without_segments=100),
            extraction_identifier=self.extraction_identifier,
        )

        train_set, test_set = pdf_to_text_extractor.get_train_test_sets(extraction_data)

        self.assertEqual(160, len(train_set.samples))
        self.assertEqual(40, len(test_set.samples))

    def test_get_train_test_only_labels_with_segments(self):
        pdf_to_text_extractor = PdfToTextExtractor(self.extraction_identifier)

        extraction_data = ExtractionData(
            samples=self.get_samples(count_with_segments=200, count_without_segments=0),
            extraction_identifier=self.extraction_identifier,
        )

        train_set, test_set = pdf_to_text_extractor.get_train_test_sets(extraction_data)

        self.assertEqual(140, len(train_set.samples))
        self.assertEqual(60, len(test_set.samples))
