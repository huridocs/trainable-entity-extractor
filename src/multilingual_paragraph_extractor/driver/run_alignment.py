import pickle
import subprocess
from pathlib import Path
from time import time

from multilingual_paragraph_extractor.domain.AlignmentScore import AlignmentScore
from multilingual_paragraph_extractor.driver.Labels import Labels

from multilingual_paragraph_extractor.domain.ParagraphsFromLanguage import ParagraphsFromLanguage
from multilingual_paragraph_extractor.driver.alignment_benchmark import save_mistakes
from multilingual_paragraph_extractor.driver.label_data import (
    get_paragraphs,
    PARAGRAPH_EXTRACTION_PATH,
    get_segmentation_data,
)
from multilingual_paragraph_extractor.use_cases.MultilingualParagraphAlignerUseCase import (
    MultilingualParagraphAlignerUseCase,
)
from trainable_entity_extractor.config import ROOT_PATH
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.use_cases.XmlFile import XmlFile

FILE_NAME = "N2432440"
EXTRACTION_IDENTIFIER = ExtractionIdentifier(extraction_name="run_alignment")
PDFS_PATH = ROOT_PATH / "data/paragraph_extraction/pdfs"


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
    pdf_data_path = ROOT_PATH / "data/paragraph_extraction/pdf_data"
    paragraphs_from_languages = list()
    for pdf_file in pdf_data_path.iterdir():
        if FILE_NAME not in pdf_file.name:
            continue

        language = pdf_file.stem.split("_")[1].split(".")[0]
        paragraph_features = get_paragraphs(pdf_file.stem)
        paragraphs_from_languages.append(
            ParagraphsFromLanguage(language=language, paragraphs=paragraph_features, is_main_language=False)
        )

    save_mistakes_paragraphs(paragraphs_from_languages)


def save_mistakes_paragraphs(paragraphs_from_languages):
    english_paragraphs = [lang for lang in paragraphs_from_languages if lang.language in ["eng", "en"]]
    if not english_paragraphs:
        return

    others = [x for x in paragraphs_from_languages if x.language != english_paragraphs[0].language]
    for other in others:
        other = other.model_copy()
        main_language = english_paragraphs[0].model_copy()
        main_language.is_main_language = True
        other.is_main_language = False
        start = time()
        rest_languages = [x for x in others if x.language != other.language]
        MultilingualParagraphAlignerUseCase(EXTRACTION_IDENTIFIER).align_languages([main_language, other] + rest_languages)
        print("align in ", round(time() - start, 2), "s")
        labels = get_labels(FILE_NAME, main_language, other)
        for label in labels:
            save_mistakes(label, label)
        break


def get_paths(pdf_name):
    xml_name = pdf_name + ".xml"
    pdf_name = pdf_name + ".pdf"
    xml_path = Path(PARAGRAPH_EXTRACTION_PATH, "xmls", xml_name)
    pdf_data_path = Path(PARAGRAPH_EXTRACTION_PATH, "pdf_data", pdf_name.replace(".pdf", ".pickle"))

    return xml_path, pdf_data_path


def create_pdf_data():
    for pdf_file in PDFS_PATH.iterdir():
        if FILE_NAME not in pdf_file.name:
            continue

        xml_path, pdf_data_path = get_paths(pdf_file.name.replace(".pdf", ".xml"))

        if not xml_path.exists():
            subprocess.run(["pdftohtml", "-i", "-xml", "-zoom", "1.0", pdf_file, xml_path])

        if pdf_data_path.exists():
            continue

        start = time()
        print("start")
        segmentation_data = get_segmentation_data(pdf_file, False)
        print("segmentation per PDF", round(time() - start, 2), "s")

        with open(xml_path, "rb") as file:
            xml_file = XmlFile(extraction_identifier=EXTRACTION_IDENTIFIER, to_train=True, xml_file_name=xml_path.name)
            xml_file.save(file_content=file.read())
        pdf_data = PdfData.from_xml_file(xml_file=xml_file, segmentation_data=segmentation_data)

        with open(pdf_data_path, "wb") as f:
            pickle.dump(pdf_data, f)


if __name__ == "__main__":
    # create_pdf_data()
    run_alignment()
