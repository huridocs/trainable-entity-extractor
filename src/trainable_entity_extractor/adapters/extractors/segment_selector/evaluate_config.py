from os.path import join

from trainable_entity_extractor.config import ROOT_PATH

METHODS_TO_EXECUTE = ["titles_history"]

LABELED_DATA_TO_USE = [
    # "rightdocs_titles",
    # "title",
    "decides",
    "signatories",
    "first_paragraph_having_seen",
    "secretary",
    "president",
    "date",
    "plan_many_date",
    "plan_many_title",
    "semantic_president",
]

SIZES = [3000]
SEED = 42

PDF_LABELED_DATA_PATH = join(ROOT_PATH.parent, "pdf-labeled-data")
