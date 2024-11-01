from pathlib import Path
from setuptools import setup

requirements_path = Path("requirements.txt")
requirements = [r for r in requirements_path.read_text().splitlines() if not r.startswith("git+")]
dependency_links = [r for r in requirements_path.read_text().splitlines() if r.startswith("git+")]
dependency_links_egg = [
    dependency_links[0] + "#egg=pdf_features",
    dependency_links[0] + "#egg=pdf_tokens_type_trainer",
    dependency_links[0] + "#egg=pdf_token_type_labels",
    dependency_links[0] + "#egg=fast_trainer",
]

PROJECT_NAME = "trainable-entity-extractor"


def get_recursive_subfolders(origin_path, recursive_path: Path):
    for sub_path in recursive_path.iterdir():
        avoid_path = False
        for text in ["test", "__pycache__", "labeled_data", "results"]:
            if text in str(sub_path):
                avoid_path = True

        if avoid_path:
            continue

        if sub_path.is_dir():
            yield from get_recursive_subfolders(origin_path, sub_path)
            yield str(sub_path).replace(str(origin_path) + "/", "").replace("/", ".")


package_path = Path(Path(__file__), "src", "trainable_entity_extractor").resolve()
base_path = Path(Path(__file__), "src")

setup(
    name=PROJECT_NAME,
    packages=[
        "trainable_entity_extractor",
        "trainable_entity_extractor.extractors",
        "trainable_entity_extractor.data",
        "trainable_entity_extractor.extractors.text_to_text_extractor.methods",
        "trainable_entity_extractor.extractors.text_to_text_extractor",
        "trainable_entity_extractor.extractors.research_multi_option_extraction",
        "trainable_entity_extractor.extractors.bert_method_scripts",
        "trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_labels_methods",
        "trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.filter_segments_methods",
        "trainable_entity_extractor.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods",
        "trainable_entity_extractor.extractors.segment_selector",
        "trainable_entity_extractor.extractors.pdf_to_multi_option_extractor",
        "trainable_entity_extractor.extractors.segment_selector.methods",
        "trainable_entity_extractor.extractors.segment_selector.methods.common_words_weights",
        "trainable_entity_extractor.extractors.segment_selector.methods.frequent_6_words",
        "trainable_entity_extractor.extractors.segment_selector.methods.next_previous_title",
        "trainable_entity_extractor.extractors.segment_selector.methods.avoiding_words",
        "trainable_entity_extractor.extractors.segment_selector.methods.best_features_10",
        "trainable_entity_extractor.extractors.segment_selector.methods.lightgbm_frequent_words",
        "trainable_entity_extractor.extractors.segment_selector.methods.best_features",
        "trainable_entity_extractor.extractors.segment_selector.methods.best_features_50",
        "trainable_entity_extractor.extractors.segment_selector.methods.base_frequent_words",
        "trainable_entity_extractor.extractors.segment_selector.methods.titles_history",
        "trainable_entity_extractor.extractors.text_to_multi_option_extractor.methods",
        "trainable_entity_extractor.extractors.text_to_multi_option_extractor",
        "trainable_entity_extractor.extractors.pdf_to_text_extractor.methods",
        "trainable_entity_extractor.extractors.pdf_to_text_extractor",
    ],
    package_dir={"": "src"},
    version="0.13",
    url="https://github.com/huridocs/trainable-entity-extractor",
    author="HURIDOCS",
    description="This tool is a trainable text/PDF to entity extractor",
    install_requires=requirements,
    setup_requires=requirements,
    dependency_links=dependency_links_egg,
)
