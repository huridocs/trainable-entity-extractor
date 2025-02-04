import pickle
from os.path import join
from fuzzywuzzy import fuzz
from fast_trainer.Paragraph import Paragraph
from fast_trainer.PdfSegment import PdfSegment

from trainable_entity_extractor.config import ROOT_PATH

label_data = {
    "cejil1": ["Eduardo Ferrer Mac-Gregor Poisot"],
    "cejil7": ["Sergio García Ramírez"],
    "cejil8": ["Héctor Fix-Zamudio"],
    "cejil_b1": ["Sergio García Ramírez"],
    "cejil_b2": ["Rafael Nieto Navia"],
    "cejil_b3": ["Thomas Buergenthal"],
    "cejil_staging1": ["Diego García-Sayán"],
    "cejil_staging10": ["Elizabeth O. Benito"],
    "cejil_staging12": ["Elizabeth O. Benito"],
    "cejil_staging15": ["Elizabeth O. Benito"],
    "cejil_staging17": ["Elizabeth O. Benito"],
    "cejil_staging2": ["Elizabeth O. Benito"],
    "cejil_staging20": ["Elizabeth O. Benito"],
    "cejil_staging21": ["Elizabeth O. Benito"],
    "cejil_staging24": ["Elizabeth O. Benito"],
    "cejil_staging27": ["Elizabeth O. Benito"],
    "cejil_staging31": ["Elizabeth O. Benito"],
    "cejil_staging4": ["Elizabeth O. Benito"],
    "cejil_staging44": ["Elizabeth O. Benito"],
    "cejil_staging54": ["Elizabeth O. Benito"],
    "cejil_staging55": ["Elizabeth O. Benito"],
    "cejil_staging56": ["Elizabeth O. Benito"],
    "cejil_staging58": ["L. Patricio Pazmiño Freire"],
    "cejil_staging66": ["Elizabeth O. Benito"],
    "cejil_staging71": ["Elizabeth O. Benito"],
    "cejil_staging72": ["Elizabeth O. Benito"],
    "cejil_staging8": ["Joel Hernández", "Esmeralda Test Arosemena Bernal de Troitiño"],
}


def load_paragraphs(pdf_name) -> list[Paragraph]:
    paragraphs_path = join(ROOT_PATH, f"data/paragraphs_cache/{pdf_name}.pickle")
    with open(paragraphs_path, mode="rb") as file:
        paragraphs: list[Paragraph] = pickle.load(file)

    return paragraphs


options = [
    "Thomas Buergenthal",
    "Héctor Fix-Zamudio",
    "Eduardo Ferrer Mac-Gregor Poisot",
    "Joel Hernández",
    "Sergio García Ramírez",
    "Rafael Nieto Navia",
    "Elizabeth O. Benito",
    "L. Patricio Pazmiño Freire",
    "Diego García-Sayán",
    "Esmeralda Test Arosemena Bernal de Troitiño",
]


if __name__ == "__main__":
    ratio_threshold = 75
    pdf_paragraphs = dict()
    for pdf_name, presidents in label_data.items():
        paragraphs = load_paragraphs(pdf_name)
        pdf_paragraphs[pdf_name] = paragraphs
        pdf_segments = [PdfSegment.from_pdf_tokens(x.tokens) for x in paragraphs]
        is_president = False
        for president in [x for x in options if x not in presidents]:
            for pdf_segment in pdf_segments:
                ratio = fuzz.ratio(president, pdf_segment.text_cleaned)
                if ratio > ratio_threshold:
                    # print(president, pdf_segment.text_content)
                    is_president = True
                    break
            if is_president:
                break

        print(pdf_name, president, is_president)

    # print('done')
    #
    # pdf_text = "Esmeralda E. Arosemena Bernal de Troitiño"
    # option = "Esmeralda Test Arosemena Bernal de Troitiño"
    # ratio = fuzz.ratio(pdf_text, option)
    # print(ratio)
    #
    # pdf_text = "Elizabeth Odio"
    # option = "Elizabeth blah Odio"
    # ratio = fuzz.ratio(pdf_text, option)
    # print(ratio)
