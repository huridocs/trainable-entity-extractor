from pathlib import Path

from multilingual_paragraph_extractor.domain.AlignmentScore import AlignmentScore
from multilingual_paragraph_extractor.driver.Labels import Labels

from multilingual_paragraph_extractor.domain.ParagraphsFromLanguage import ParagraphsFromLanguage
from multilingual_paragraph_extractor.driver.alignment_benchmark import save_mistakes
from multilingual_paragraph_extractor.driver.label_data import get_paragraphs
from multilingual_paragraph_extractor.use_cases.MultilingualParagraphAlignerUseCase import (
    MultilingualParagraphAlignerUseCase,
)
from trainable_entity_extractor.config import ROOT_PATH
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier

FILE_NAME = "ba"
EXTRACTION_IDENTIFIER = ExtractionIdentifier(extraction_name="run_alignment")


def get_labels(
    pdf_name: str, main_paragraphs: ParagraphsFromLanguage, other_paragraphs: ParagraphsFromLanguage
) -> list[Labels]:
    labels_list: list[Labels] = list()

    MultilingualParagraphAlignerUseCase(EXTRACTION_IDENTIFIER).align_languages([main_paragraphs, other_paragraphs])
    label = Labels(
        main_language=main_paragraphs.language,
        other_language=other_paragraphs.language,
        main_xml_name=pdf_name + "_" + main_paragraphs.language + ".xml",
        other_xml_name=pdf_name + "_" + other_paragraphs.language + ".xml",
    )
    for main_paragraph, other_paragraphs in zip(main_paragraphs.paragraphs, other_paragraphs.paragraphs):
        label.add_paragraph(AlignmentScore(main_paragraph=main_paragraph, other_paragraph=other_paragraphs, score=1))

    labels_list.append(label)

    return labels_list


def run_alignment():
    pdf_data_path = Path(ROOT_PATH, "data/paragraph_extraction/pdf_data")
    paragraphs_from_languages = list()
    for pdf_file in pdf_data_path.iterdir():
        if FILE_NAME not in pdf_file.name:
            continue
        language = pdf_file.stem.split("_")[1]
        paragraph_features = get_paragraphs(pdf_file.stem)
        paragraphs_from_languages.append(
            ParagraphsFromLanguage(language=language, paragraphs=paragraph_features, is_main_language=False)
        )

    save_mistakes_paragraphs(paragraphs_from_languages)


def save_mistakes_paragraphs(paragraphs_from_languages):
    for paragraphs_from_language in paragraphs_from_languages:
        main_language = paragraphs_from_language.model_copy()
        main_language.is_main_language = True
        other = [x for x in paragraphs_from_languages if x != paragraphs_from_language][0].model_copy()
        other.is_main_language = False
        MultilingualParagraphAlignerUseCase(EXTRACTION_IDENTIFIER).align_languages([main_language, other])
        labels = get_labels(FILE_NAME, main_language, other)
        # for label in labels:
        #     save_mistakes(label, label)


if __name__ == "__main__":
    run_alignment()
