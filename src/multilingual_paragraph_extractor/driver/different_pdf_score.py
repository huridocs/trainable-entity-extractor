from multilingual_paragraph_extractor.domain.ParagraphsFromLanguage import ParagraphsFromLanguage
from multilingual_paragraph_extractor.driver.Labels import Labels
from multilingual_paragraph_extractor.driver.alignment_benchmark import save_mistakes
from multilingual_paragraph_extractor.driver.label_data import get_paragraphs, EXTRACTION_IDENTIFIER
from multilingual_paragraph_extractor.use_cases.MultilingualParagraphAlignerUseCase import (
    MultilingualParagraphAlignerUseCase,
)


def different_pdf_score(pdf_name_1: str, pdf_name_2: str):
    main_language = pdf_name_1.split("_")[-1]
    other_language = pdf_name_2.split("_")[-1]

    main_paragraphs = ParagraphsFromLanguage(
        language=main_language, paragraphs=get_paragraphs(pdf_name_1), is_main_language=True
    )
    other_paragraphs = ParagraphsFromLanguage(
        language=other_language, paragraphs=get_paragraphs(pdf_name_2), is_main_language=False
    )

    MultilingualParagraphAlignerUseCase(EXTRACTION_IDENTIFIER).align_languages([main_paragraphs, other_paragraphs])

    label = get_label(main_paragraphs, other_paragraphs, pdf_name_1)
    save_mistakes(label, label)
    match_percentage = 100 * (
        1 - (len(main_paragraphs.paragraphs) - len(label.paragraphs)) / len(main_paragraphs.paragraphs)
    )
    print(f"Match percentage: {round(match_percentage, 2)}%")

    print("ok")


def get_label(main_paragraphs, other_paragraphs, pdf_name_1):
    label = Labels(
        main_language="es",
        other_language="different_pdf",
        main_xml_name=pdf_name_1 + ".xml",
        other_xml_name=pdf_name_1 + ".xml",
    )

    for paragraph in main_paragraphs.paragraphs:
        if paragraph not in other_paragraphs._alignment_scores:
            continue
        alignment_score = other_paragraphs._alignment_scores[paragraph]
        label.add_paragraph(alignment_score)
    return label


if __name__ == "__main__":
    different_pdf_score("1dcf1hho0p6_eng", "1dcf1hho0p6_fra")
