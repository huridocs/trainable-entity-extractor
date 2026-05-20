from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.drivers.TrainableEntityExtractor import TrainableEntityExtractor


class TestSanitizeLanguages(TestCase):
    def setUp(self):
        self.extraction_identifier = ExtractionIdentifier(run_name="unit_test", extraction_name="sanitize_test")

    def test_valid_two_letter_code_unchanged(self):
        sample = TrainingSample(labeled_data=LabeledData(language_iso="en"))
        extraction_data = ExtractionData(samples=[sample], extraction_identifier=self.extraction_identifier)
        TrainableEntityExtractor._sanitize_languages(extraction_data)
        self.assertEqual("en", extraction_data.samples[0].labeled_data.language_iso)

    def test_valid_script_variant_unchanged(self):
        sample = TrainingSample(labeled_data=LabeledData(language_iso="zh-Hans"))
        extraction_data = ExtractionData(samples=[sample], extraction_identifier=self.extraction_identifier)
        TrainableEntityExtractor._sanitize_languages(extraction_data)
        self.assertEqual("zh-Hans", extraction_data.samples[0].labeled_data.language_iso)

    def test_three_letter_to_two_letter_eng(self):
        sample = TrainingSample(labeled_data=LabeledData(language_iso="eng"))
        extraction_data = ExtractionData(samples=[sample], extraction_identifier=self.extraction_identifier)
        TrainableEntityExtractor._sanitize_languages(extraction_data)
        self.assertEqual("en", extraction_data.samples[0].labeled_data.language_iso)

    def test_three_letter_to_two_letter_spa(self):
        sample = TrainingSample(labeled_data=LabeledData(language_iso="spa"))
        extraction_data = ExtractionData(samples=[sample], extraction_identifier=self.extraction_identifier)
        TrainableEntityExtractor._sanitize_languages(extraction_data)
        self.assertEqual("es", extraction_data.samples[0].labeled_data.language_iso)

    def test_three_letter_to_two_letter_fra(self):
        sample = TrainingSample(labeled_data=LabeledData(language_iso="fra"))
        extraction_data = ExtractionData(samples=[sample], extraction_identifier=self.extraction_identifier)
        TrainableEntityExtractor._sanitize_languages(extraction_data)
        self.assertEqual("fr", extraction_data.samples[0].labeled_data.language_iso)

    def test_three_letter_to_two_letter_deu(self):
        sample = TrainingSample(labeled_data=LabeledData(language_iso="deu"))
        extraction_data = ExtractionData(samples=[sample], extraction_identifier=self.extraction_identifier)
        TrainableEntityExtractor._sanitize_languages(extraction_data)
        self.assertEqual("de", extraction_data.samples[0].labeled_data.language_iso)

    def test_invalid_code_other_defaults_to_en(self):
        sample = TrainingSample(labeled_data=LabeledData(language_iso="other"))
        extraction_data = ExtractionData(samples=[sample], extraction_identifier=self.extraction_identifier)
        TrainableEntityExtractor._sanitize_languages(extraction_data)
        self.assertEqual("en", extraction_data.samples[0].labeled_data.language_iso)

    def test_invalid_code_unknown_defaults_to_en(self):
        sample = TrainingSample(labeled_data=LabeledData(language_iso="unknown"))
        extraction_data = ExtractionData(samples=[sample], extraction_identifier=self.extraction_identifier)
        TrainableEntityExtractor._sanitize_languages(extraction_data)
        self.assertEqual("en", extraction_data.samples[0].labeled_data.language_iso)

    def test_empty_string_defaults_to_en(self):
        sample = TrainingSample(labeled_data=LabeledData(language_iso=""))
        extraction_data = ExtractionData(samples=[sample], extraction_identifier=self.extraction_identifier)
        TrainableEntityExtractor._sanitize_languages(extraction_data)
        self.assertEqual("en", extraction_data.samples[0].labeled_data.language_iso)

    def test_mixed_codes_in_multiple_samples(self):
        samples = [
            TrainingSample(labeled_data=LabeledData(language_iso="en")),
            TrainingSample(labeled_data=LabeledData(language_iso="eng")),
            TrainingSample(labeled_data=LabeledData(language_iso="spa")),
            TrainingSample(labeled_data=LabeledData(language_iso="other")),
            TrainingSample(labeled_data=LabeledData(language_iso="")),
            TrainingSample(labeled_data=LabeledData(language_iso="zh-Hans")),
        ]
        extraction_data = ExtractionData(samples=samples, extraction_identifier=self.extraction_identifier)
        TrainableEntityExtractor._sanitize_languages(extraction_data)
        self.assertEqual("en", extraction_data.samples[0].labeled_data.language_iso)
        self.assertEqual("en", extraction_data.samples[1].labeled_data.language_iso)
        self.assertEqual("es", extraction_data.samples[2].labeled_data.language_iso)
        self.assertEqual("en", extraction_data.samples[3].labeled_data.language_iso)
        self.assertEqual("en", extraction_data.samples[4].labeled_data.language_iso)
        self.assertEqual("zh-Hans", extraction_data.samples[5].labeled_data.language_iso)

    def test_code_without_two_letter_mapping_defaults_to_en(self):
        sample = TrainingSample(labeled_data=LabeledData(language_iso="cmn"))
        extraction_data = ExtractionData(samples=[sample], extraction_identifier=self.extraction_identifier)
        TrainableEntityExtractor._sanitize_languages(extraction_data)
        self.assertEqual("en", extraction_data.samples[0].labeled_data.language_iso)

    def test_spanish_code_es_unchanged(self):
        sample = TrainingSample(labeled_data=LabeledData(language_iso="es"))
        extraction_data = ExtractionData(samples=[sample], extraction_identifier=self.extraction_identifier)
        TrainableEntityExtractor._sanitize_languages(extraction_data)
        self.assertEqual("es", extraction_data.samples[0].labeled_data.language_iso)

    def test_code_az_Cyrl_preserved(self):
        sample = TrainingSample(labeled_data=LabeledData(language_iso="az-Cyrl"))
        extraction_data = ExtractionData(samples=[sample], extraction_identifier=self.extraction_identifier)
        TrainableEntityExtractor._sanitize_languages(extraction_data)
        self.assertEqual("az-Cyrl", extraction_data.samples[0].labeled_data.language_iso)
