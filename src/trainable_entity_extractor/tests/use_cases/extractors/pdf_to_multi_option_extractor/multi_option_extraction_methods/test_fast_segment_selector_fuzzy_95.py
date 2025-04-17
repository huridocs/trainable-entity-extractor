from time import time
from unittest import TestCase
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FastSegmentSelectorFuzzy95 import (
    FastSegmentSelectorFuzzy95,
)
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FastSegmentSelectorFuzzyCommas import (
    FastSegmentSelectorFuzzyCommas,
)
from trainable_entity_extractor.use_cases.extractors.segment_selector.FastSegmentSelector import FastSegmentSelector


class TestFastSegmentSelectorFuzzy95(TestCase):
    TENANT = "multi_option_extraction_test"
    extraction_id = "extraction_id"
    extraction_identifier = ExtractionIdentifier(run_name=TENANT, extraction_name=extraction_id)

    def setUp(self):
        FastSegmentSelector(self.extraction_identifier).prepare_model_folder()

    def test_performance(self):
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        pdf_data_1 = PdfData.from_texts(["1, 2"])
        pdf_data_2 = PdfData.from_texts(["2"])
        pdf_data_3 = PdfData.from_texts(["3, 1"])
        pdf_data_4 = PdfData.from_texts(["2, 3"])
        pdf_data_5 = PdfData.from_texts(
            [
                """86.
Antigua-et-Barbuda a pris en compte les recommandations relatives à la création
d’une institution nationale des droits de l’homme conforme aux Principes concernant le statut
des institutions nationales pour la promotion et la protection des droits de l’homme (Principes
de Paris). Elle a admis qu’elle ne disposait pas d’un mécanisme centralisé de signalement des
violations des droits de l’homme ou d’un système centralisé de collecte de statistiques. Elle
avait conscience de l’importance de se doter d’une telle institution afin d’être en mesure
d’adresser des signalements aux organismes internationaux et de diffuser un enseignement et
des informations sur la promotion et la protection des droits de l’homme à Antigua-et-
Barbuda."""
                for _ in range(3000)
            ]
        )

        samples = [
            TrainingSample(pdf_data=pdf_data_1, labeled_data=LabeledData(values=[options[0], options[1]])),
            TrainingSample(pdf_data=pdf_data_2, labeled_data=LabeledData(values=[options[1]])),
            TrainingSample(pdf_data=pdf_data_3, labeled_data=LabeledData(values=[options[2], options[0]])),
            TrainingSample(pdf_data=pdf_data_4, labeled_data=LabeledData(values=[options[1], options[2]])),
            TrainingSample(pdf_data=pdf_data_5, labeled_data=LabeledData(values=[])),
        ]

        multi_option_data = ExtractionData(
            multi_value=True, options=options, samples=samples, extraction_identifier=self.extraction_identifier
        )

        fast_segment_selector_fuzzy_95 = FastSegmentSelectorFuzzy95()

        start = time()
        print("start")
        performance = fast_segment_selector_fuzzy_95.get_performance(multi_option_data, multi_option_data)
        print("time", round(time() - start, 2), "s")

        self.assertEqual(0, performance)
