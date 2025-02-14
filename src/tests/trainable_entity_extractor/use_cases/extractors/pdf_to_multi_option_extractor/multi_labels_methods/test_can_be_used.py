from unittest import TestCase

import torch

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.multi_labels_methods.SetFitEnglishMethod import (
    SetFitEnglishMethod,
)
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.multi_labels_methods.SetFitMultilingualMethod import (
    SetFitMultilingualMethod,
)
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.multi_labels_methods.SingleLabelSetFitEnglishMethod import (
    SingleLabelSetFitEnglishMethod,
)
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.multi_labels_methods.SingleLabelSetFitMultilingualMethod import (
    SingleLabelSetFitMultilingualMethod,
)


class TestSetFitEnglishMethod(TestCase):
    TENANT = "multi_option_extraction_test"
    extraction_id = "extraction_id"

    def setUp(self):
        self.extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        self.options = [
            Option(id="1", label="1"),
            Option(id="2", label="2"),
            Option(id="3", label="3"),
            Option(id="4", label="4"),
            Option(id="5", label="5"),
            Option(id="6", label="6"),
            Option(id="7", label="7"),
            Option(id="8", label="8"),
        ]

        pdf_data_1 = PdfData.from_texts(["point 1"])
        pdf_data_2 = PdfData.from_texts(["point 2 point 3"])
        pdf_data_3 = PdfData.from_texts(["point 3 point 4"])
        pdf_data_4 = PdfData.from_texts(["point 4"])
        pdf_data_5 = PdfData.from_texts(["point 5 point 1"])
        pdf_data_6 = PdfData.from_texts(["point 6"])
        pdf_data_7 = PdfData.from_texts(["point 7 point 6"])
        pdf_data_8 = PdfData.from_texts(["point 8"])

        samples = [
            TrainingSample(pdf_data_1, LabeledData(values=[self.options[0]], language_iso="en")),
            TrainingSample(pdf_data_2, LabeledData(values=[self.options[1], self.options[2]], language_iso="es")),
            TrainingSample(pdf_data_3, LabeledData(values=[self.options[2], self.options[3]], language_iso="en")),
            TrainingSample(pdf_data_4, LabeledData(values=[self.options[3]], language_iso="fr")),
            TrainingSample(pdf_data_5, LabeledData(values=[self.options[4], self.options[0]], language_iso="en")),
            TrainingSample(pdf_data_6, LabeledData(values=[self.options[5]], language_iso="en")),
            TrainingSample(pdf_data_7, LabeledData(values=[self.options[6], self.options[5]], language_iso="ru")),
            TrainingSample(pdf_data_8, LabeledData(values=[self.options[7]], language_iso="ru")),
        ]

        self.extraction_data_english_multi = ExtractionData(
            multi_value=True,
            options=self.options,
            samples=[samples[2], samples[4]],
            extraction_identifier=self.extraction_identifier,
        )

        self.extraction_data_english_single = ExtractionData(
            multi_value=False,
            options=self.options,
            samples=[samples[0], samples[5]],
            extraction_identifier=self.extraction_identifier,
        )

        self.extraction_data_non_english_multi = ExtractionData(
            multi_value=True,
            options=self.options,
            samples=[samples[1], samples[6]],
            extraction_identifier=self.extraction_identifier,
        )

        self.extraction_data_non_english_single = ExtractionData(
            multi_value=False,
            options=self.options,
            samples=[samples[3], samples[7]],
            extraction_identifier=self.extraction_identifier,
        )

    def test_can_be_used_english_single_label(self):
        if not torch.cuda.is_available():
            return
        setfit_single_english_method = SingleLabelSetFitEnglishMethod(self.extraction_identifier, self.options, False)
        self.assertFalse(setfit_single_english_method.can_be_used(self.extraction_data_english_multi))
        self.assertTrue(setfit_single_english_method.can_be_used(self.extraction_data_english_single))
        self.assertFalse(setfit_single_english_method.can_be_used(self.extraction_data_non_english_multi))
        self.assertFalse(setfit_single_english_method.can_be_used(self.extraction_data_non_english_single))

    def test_can_be_used_english_multi_label(self):
        if not torch.cuda.is_available():
            return
        setfit_english_method = SetFitEnglishMethod(self.extraction_identifier, self.options, True)
        self.assertTrue(setfit_english_method.can_be_used(self.extraction_data_english_multi))
        self.assertFalse(setfit_english_method.can_be_used(self.extraction_data_english_single))
        self.assertFalse(setfit_english_method.can_be_used(self.extraction_data_non_english_multi))
        self.assertFalse(setfit_english_method.can_be_used(self.extraction_data_non_english_single))

    def test_can_be_used_non_english_single_label(self):
        if not torch.cuda.is_available():
            return
        setfit_single_multilingual_method = SingleLabelSetFitMultilingualMethod(
            self.extraction_identifier, self.options, False
        )
        self.assertFalse(setfit_single_multilingual_method.can_be_used(self.extraction_data_english_multi))
        self.assertFalse(setfit_single_multilingual_method.can_be_used(self.extraction_data_english_single))
        self.assertFalse(setfit_single_multilingual_method.can_be_used(self.extraction_data_non_english_multi))
        self.assertTrue(setfit_single_multilingual_method.can_be_used(self.extraction_data_non_english_single))

    def test_can_be_used_non_english_multi_label(self):
        if not torch.cuda.is_available():
            return
        setfit_multilingual_method = SetFitMultilingualMethod(self.extraction_identifier, self.options, True)
        self.assertFalse(setfit_multilingual_method.can_be_used(self.extraction_data_english_multi))
        self.assertFalse(setfit_multilingual_method.can_be_used(self.extraction_data_english_single))
        self.assertTrue(setfit_multilingual_method.can_be_used(self.extraction_data_non_english_multi))
        self.assertFalse(setfit_multilingual_method.can_be_used(self.extraction_data_non_english_single))
