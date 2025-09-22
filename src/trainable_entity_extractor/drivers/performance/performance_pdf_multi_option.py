from time import time

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.PdfMultiOptionMethod import (
    PdfMultiOptionMethod,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FastSegmentSelectorFuzzy95 import (
    FastSegmentSelectorFuzzy95,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FastSegmentSelectorFuzzyCommas import (
    FastSegmentSelectorFuzzyCommas,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzySegmentSelector import (
    FuzzySegmentSelector,
)

TENANT = "performance_pdf_multi_option"
extraction_id = "performance_pdf_multi_option"
extraction_identifier = ExtractionIdentifier(run_name=TENANT, extraction_name=extraction_id)


def run():
    methods: list[PdfMultiOptionMethod] = [
        # FuzzyFirst(),
        # FuzzyLast(),
        # FuzzyFirstCleanLabel(),
        # FuzzyLastCleanLabel(),
        # FuzzyAll100(),
        # FuzzyAll88(),
        # FuzzyAll75(),
        # PreviousWordsTokenSelectorFuzzy75(),
        # NextWordsTokenSelectorFuzzy75(),
        # PreviousWordsSentenceSelectorFuzzyCommas(),
        FastSegmentSelectorFuzzy95(),
        FastSegmentSelectorFuzzyCommas(),
        FuzzySegmentSelector(),
        # PdfMultiOptionMethod(CleanBeginningDotDigits500, FastTextMethod),
        # PdfMultiOptionMethod(CleanEndDotDigits1000, FastTextMethod),
        # PdfMultiOptionMethod(CleanBeginningDot1000, SetFitEnglishMethod),
        # PdfMultiOptionMethod(CleanBeginningDot1000, SetFitMultilingualMethod),
        # PdfMultiOptionMethod(CleanBeginningDot1000, SingleLabelSetFitEnglishMethod),
        # PdfMultiOptionMethod(CleanBeginningDot1000, SingleLabelSetFitMultilingualMethod),
    ]

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
            for _ in range(3000 // 10)
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
        multi_value=True, options=options, samples=samples, extraction_identifier=extraction_identifier
    )

    times = ""
    for method in methods:
        print(f"Testing method: {method.get_name()}")
        start = time()
        method.get_performance(multi_option_data, multi_option_data)
        print("total time", round(time() - start, 2), "s")
        times += f"{method.get_name()}: {round(time() - start, 2)}\n"

    print(times)


if __name__ == "__main__":
    run()
