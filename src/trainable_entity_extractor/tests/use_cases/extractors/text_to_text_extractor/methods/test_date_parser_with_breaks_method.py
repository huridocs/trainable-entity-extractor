from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.DateParserWithBreaksMethod import (
    DateParserWithBreaksMethod,
)


class TestDateParserWithBreaksMethod(TestCase):
    def test_predict(self):
        extraction_identifier = ExtractionIdentifier(run_name="test", extraction_name="test")

        sample = TrainingSample(
            labeled_data=LabeledData(label_text="1982-06-05", language_iso="en"), segment_selector_texts=["5 Jun 1982"]
        )

        extraction_data = ExtractionData(samples=[sample for _ in range(6)], extraction_identifier=extraction_identifier)
        date_parser_method = DateParserWithBreaksMethod(extraction_identifier)

        date_parser_method.train(extraction_data)

        text_1 = ["ORDER OF THE INTER-AMERICAN COURT OF HUMAN RIGHTS 1", "OF FEBRUARY 9, 2006"]
        text_2 = ["ORDER OF THE INTER-AMERICAN COURT OF HUMAN RIGHTS 1", "OF MARCH 10, 2007"]
        predictions = date_parser_method.predict([PredictionSample.from_texts(text_1), PredictionSample.from_texts(text_2)])

        self.assertEqual("2006-02-09", predictions[0])
        self.assertEqual("2007-03-10", predictions[1])
