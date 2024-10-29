from pathlib import Path

from setuptools import setup


requirements_path = Path("requirements.txt")
requirements = [r for r in requirements_path.read_text().splitlines() if not r.startswith("git+")]
dependency_links = [r for r in requirements_path.read_text().splitlines() if r.startswith("git+")]

PROJECT_NAME = "trainable-entity-extractor"

setup(
    name=PROJECT_NAME,
    packages=["trainable_entity_extractor"],
    package_dir={"": "src"},
    version="0.1",
    url="https://github.com/huridocs/trainable-entity-extractor",
    author="HURIDOCS",
    description="This tool is a trainable text/PDF to entity extractor",
    install_requires=requirements,
    setup_requires=requirements,
    dependency_links=dependency_links,
)
