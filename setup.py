from pathlib import Path
from setuptools import setup

requirements_path = Path("requirements.txt")
requirements = [r for r in requirements_path.read_text().splitlines() if not r.startswith("git+")]
dependency_links = [r for r in requirements_path.read_text().splitlines() if r.startswith("git+")]

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
    packages=["trainable_entity_extractor"] + [folder for folder in get_recursive_subfolders(base_path, package_path)],
    package_dir={"": "src"},
    version="0.8",
    url="https://github.com/huridocs/trainable-entity-extractor",
    author="HURIDOCS",
    description="This tool is a trainable text/PDF to entity extractor",
    install_requires=requirements,
    setup_requires=requirements,
    dependency_links=dependency_links,
)
