import os
from os.path import join
from pathlib import Path

from fast_trainer.PdfSegment import PdfSegment
from pdf_token_type_labels.TokenType import TokenType
from tqdm import tqdm

from trainable_entity_extractor.config import ROOT_PATH
from pdf_topic_classification.clean_data import remove_one_line_paragraph
from pdf_topic_classification.pdf_topic_classification_data import get_labeled_data
from transformers import pipeline

# summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device="cuda:0")
summarizer = pipeline("summarization", model="Falconsai/text_summarization", device="cuda:0")

valid_types = [TokenType.TITLE, TokenType.TEXT]


def clean_content_pdf_token(texts):
    all_text = " ".join(texts)
    all_text_words = all_text.split()
    clean_words = list()
    for word in all_text_words:
        clean_word = "".join([x for x in word if x.isalpha()])
        if clean_word:
            clean_words.append(clean_word)
    return clean_words


def get_text(texts: list[str]) -> str:
    texts = clean_content_pdf_token(texts)
    total_text = ""
    final_texts: list[str] = list()
    for text in texts:
        if len(total_text + " " + text) > 1000:
            break

        total_text += " " + text
        final_texts.append(text)

    return " ".join(final_texts)


def get_summaries():
    task_labeled_data = get_labeled_data("cyrilla")[0]
    for pdf_labels in tqdm(task_labeled_data.pdfs_labels):
        path = Path(join(ROOT_PATH, "data", "summaries", f"{pdf_labels.pdf_name}.txt"))
        os.makedirs(path.parent, exist_ok=True)
        multi_line_paragraphs = remove_one_line_paragraph(pdf_labels.paragraphs)
        pdf_segments = [PdfSegment.from_pdf_tokens(paragraph.tokens) for paragraph in multi_line_paragraphs]
        pdf_segments = [x for x in pdf_segments if x.paragraph_type in valid_types]
        texts = [pdf_segment.text_content for pdf_segment in pdf_segments if pdf_segment.paragraph_type in valid_types]
        summary = summarizer(get_text(texts), max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
        path.write_text(summary)
        print(path)


if __name__ == "__main__":
    get_summaries()
