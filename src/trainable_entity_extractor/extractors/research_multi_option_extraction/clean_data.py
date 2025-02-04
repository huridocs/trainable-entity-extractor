import pickle
from os.path import join
from pathlib import Path

from fast_trainer.Paragraph import Paragraph
from fast_trainer.PdfSegment import PdfSegment
from pdf_token_type_labels.TaskMistakes import TaskMistakes
from pdf_token_type_labels.TokenType import TokenType

from trainable_entity_extractor.config import ROOT_PATH


def is_multi_line(paragraph: Paragraph):
    if not paragraph.tokens:
        return False

    if paragraph.tokens[0].token_type == TokenType.TITLE:
        return True

    paragraph_multi_line = paragraph.tokens[0].bounding_box.bottom < paragraph.tokens[-1].bounding_box.top
    return paragraph_multi_line


def remove_one_line_paragraph(paragraphs: list[Paragraph]):
    multi_line_paragraphs = list()
    for paragraph in paragraphs:
        if is_multi_line(paragraph):
            multi_line_paragraphs.append(paragraph)

    return multi_line_paragraphs


def character_density():
    with open(join(ROOT_PATH, "data/paragraphs_cache/cyrilla_58.pickle"), mode="rb") as file:
        paragraphs: list[Paragraph] = pickle.load(file)

    pdf_labeled_data_root_path = join(Path(ROOT_PATH).parent, "pdf-labeled-data")

    task_mistakes = TaskMistakes(
        pdf_labeled_data_root_path=pdf_labeled_data_root_path, test_id="characeter_density", pdf_name="cyrilla_58"
    )
    for paragraph in paragraphs:
        segment_top = min([x.bounding_box.top for x in paragraph.tokens]) + 8
        first_line_segment = PdfSegment.from_pdf_tokens([x for x in paragraph.tokens if x.bounding_box.top < segment_top])
        pdf_segment = PdfSegment.from_pdf_tokens(paragraph.tokens)
        characters_count = len([x for x in first_line_segment.text_cleaned if x.isalpha()])

        character_ratio: float = pdf_segment.bounding_box.width / characters_count if characters_count else 0

        task_mistakes.add(pdf_segment.page_number, pdf_segment.bounding_box, 1, 1, str(round(character_ratio, 1)))

    task_mistakes.save()


if __name__ == "__main__":
    character_density()
