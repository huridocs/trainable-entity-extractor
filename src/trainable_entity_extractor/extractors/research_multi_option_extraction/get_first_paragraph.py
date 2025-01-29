from fast_trainer.PdfSegment import PdfSegment
from pdf_token_type_labels.TokenType import TokenType

from pdf_topic_classification.pdf_topic_classification_data import get_labeled_data

valid_types = [TokenType.TITLE]


def get_first_paragraph():
    task_labeled_data = get_labeled_data("cyrilla")[0]
    for pdf_labels in task_labeled_data.pdfs_labels[3:]:
        pdf_segments = [PdfSegment.from_pdf_tokens(paragraph.tokens) for paragraph in pdf_labels.paragraphs]
        pdf_segments = [x for x in pdf_segments if x.paragraph_type in valid_types]
        print(pdf_labels.pdf_name)
        print([len(x.text_content) for x in pdf_segments])
        print("\n".join([x.text_content for x in pdf_segments]))
        break


if __name__ == "__main__":
    get_first_paragraph()
