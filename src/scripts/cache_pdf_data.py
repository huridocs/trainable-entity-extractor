import os
import pickle

from os.path import join, exists
from pathlib import Path

from fast_trainer.Paragraph import Paragraph
from fast_trainer.ParagraphExtractorTrainer import ParagraphExtractorTrainer
from fast_trainer.download_models import paragraph_extraction_model_path
from fast_trainer.model_configuration import MODEL_CONFIGURATION
from pdf_features.PdfFeatures import PdfFeatures
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
from pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer

from trainable_entity_extractor.config import ROOT_PATH
from trainable_entity_extractor.data.PdfData import PdfData
from performance_report import get_task_pdf_names, PDF_DATA_FOLDER_PATH

LABELED_DATA_PDFS_PATH = join(ROOT_PATH.parent, "pdf-labeled-data", "pdfs")


def cache_pdf_data():
    task_pdf_names = get_task_pdf_names()
    for task, pdf_names in task_pdf_names.items():
        if task != "countries_in_favor":
            continue

        for pdf_name in pdf_names:
            pdf_data_pickle_path = Path(str(join(PDF_DATA_FOLDER_PATH, pdf_name + ".pickle")))
            os.makedirs(pdf_data_pickle_path.parent, exist_ok=True)

            if exists(pdf_data_pickle_path):
                continue

            print("caching ", pdf_name)
            pdf_features = PdfFeatures.from_poppler_etree(join(LABELED_DATA_PDFS_PATH, pdf_name, "etree.xml"))

            trainer = TokenTypeTrainer([pdf_features], ModelConfiguration())
            trainer.set_token_types()
            trainer = ParagraphExtractorTrainer(pdfs_features=[pdf_features], model_configuration=MODEL_CONFIGURATION)
            paragraphs: list[Paragraph] = trainer.get_paragraphs(paragraph_extraction_model_path)

            pdf_data = PdfData(pdf_features, file_name=pdf_name)
            pdf_data.set_segments_from_paragraphs(paragraphs)

            with open(pdf_data_pickle_path, mode="wb") as file:
                pickle.dump(pdf_data, file)


if __name__ == "__main__":
    cache_pdf_data()
