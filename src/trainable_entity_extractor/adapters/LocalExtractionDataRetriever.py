import pickle
from typing import Optional
from pathlib import Path
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.config import CACHE_PATH
from trainable_entity_extractor.ports.ExtractionDataRetriever import ExtractionDataRetriever


class LocalExtractionDataRetriever(ExtractionDataRetriever):

    def __init__(self):
        self.cache_base_path = CACHE_PATH
        self.cache_base_path.mkdir(parents=True, exist_ok=True)

    def get_extraction_data(self, extraction_identifier: ExtractionIdentifier) -> Optional[ExtractionData]:
        cached_data = self._get_from_cache(extraction_identifier)
        if cached_data:
            return cached_data

        return None

    def save_extraction_data(self, extraction_identifier: ExtractionIdentifier, extraction_data: ExtractionData) -> bool:
        try:
            cache_path = self._get_cache_path(extraction_identifier)
            cache_path.mkdir(parents=True, exist_ok=True)

            pickle_file = cache_path / "extraction_data.pickle"
            with open(pickle_file, "wb") as f:
                pickle.dump(extraction_data, f)

            return True
        except Exception as e:
            print(f"Failed to cache extraction data: {e}")
            return False

    def save_prediction_data(
        self, extraction_identifier: ExtractionIdentifier, prediction_data: list[PredictionSample]
    ) -> bool:
        try:
            cache_path = self._get_cache_path(extraction_identifier)
            cache_path.mkdir(parents=True, exist_ok=True)

            pickle_file = cache_path / "prediction_data.pickle"
            with open(pickle_file, "wb") as f:
                pickle.dump(prediction_data, f)

            return True
        except Exception as e:
            print(f"Failed to cache prediction data: {e}")
            return False

    def get_prediction_data(self, extraction_identifier: ExtractionIdentifier) -> list[PredictionSample]:
        try:
            cache_path = self._get_cache_path(extraction_identifier)
            pickle_file = cache_path / "prediction_data.pickle"

            if pickle_file.exists():
                with open(pickle_file, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Failed to load cached prediction data: {e}")

        return []

    def _get_from_cache(self, extraction_identifier: ExtractionIdentifier) -> Optional[ExtractionData]:
        try:
            cache_path = self._get_cache_path(extraction_identifier)
            pickle_file = cache_path / "extraction_data.pickle"

            if pickle_file.exists():
                with open(pickle_file, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Failed to load cached extraction data: {e}")

        return None

    def _get_cache_path(self, extraction_identifier: ExtractionIdentifier) -> Path:
        return self.cache_base_path / extraction_identifier.run_name / extraction_identifier.extraction_name

    def get_suggestions(self, extraction_identifier: ExtractionIdentifier) -> list[Suggestion]:
        try:
            cache_path = self._get_cache_path(extraction_identifier)
            pickle_file = cache_path / "suggestions_data.pickle"

            if pickle_file.exists():
                with open(pickle_file, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Failed to load cached suggestions data: {e}")

        return []

    def save_suggestions(self, extraction_identifier: ExtractionIdentifier, suggestions: list[Suggestion]) -> bool:
        try:
            cache_path = self._get_cache_path(extraction_identifier)
            cache_path.mkdir(parents=True, exist_ok=True)

            pickle_file = cache_path / "suggestions_data.pickle"
            with open(pickle_file, "wb") as f:
                pickle.dump(suggestions, f)

            return True
        except Exception as e:
            print(f"Failed to cache suggestions data: {e}")
            return False
