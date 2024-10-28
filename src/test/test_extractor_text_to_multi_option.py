import shutil
from os.path import join
from unittest import TestCase

import mongomock
import pymongo

from trainable_entity_extractor.Extractor import TrainableEntityExtractor
from trainable_entity_extractor.config import DATA_PATH
from trainable_entity_extractor.data.ExtractionTask import ExtractionTask
from trainable_entity_extractor.data.Option import Option
from trainable_entity_extractor.data.Params import Params
from trainable_entity_extractor.data.Suggestion import Suggestion


class TestExtractorTextToMultiOption(TestCase):
    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_text_multi_option_suggestions(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "tenant_to_be_removed"
        extraction_id = "extraction_id"

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

        options = [Option(id="1", label="abc"), Option(id="2", label="dfg"), Option(id="3", label="hij")]

        labeled_data_json = {
            "id": extraction_id,
            "tenant": tenant,
            "language_iso": "en",
            "source_text": "abc dfg",
            "values": [{"id": "1", "label": "abc"}, {"id": "2", "label": "dfg"}],
        }

        mongo_client.pdf_metadata_extraction.labeled_data.insert_one(labeled_data_json)

        TrainableEntityExtractor.calculate_task(
            ExtractionTask(
                tenant=tenant,
                task=TrainableEntityExtractor.CREATE_MODEL_TASK_NAME,
                params=Params(id=extraction_id, options=options, multi_value=True),
            )
        )

        to_predict_json = [
            {
                "tenant": tenant,
                "id": extraction_id,
                "entity_name": "entity_name_2",
                "source_text": "foo var dfg hij foo var",
            }
        ]

        mongo_client.pdf_metadata_extraction.prediction_data.insert_many(to_predict_json)

        task_calculated, error = TrainableEntityExtractor.calculate_task(
            ExtractionTask(
                tenant=tenant,
                task=TrainableEntityExtractor.SUGGESTIONS_TASK_NAME,
                params=Params(id=extraction_id),
            )
        )

        suggestions: list[Suggestion] = list()
        find_filter = {"id": extraction_id, "tenant": tenant}
        for document in mongo_client.pdf_metadata_extraction.suggestions.find(find_filter):
            suggestions.append(Suggestion(**document))

        self.assertTrue(task_calculated)
        self.assertEqual(1, len(suggestions))
        self.assertEqual(tenant, suggestions[0].tenant)
        self.assertEqual(extraction_id, suggestions[0].id)
        self.assertEqual("entity_name_2", suggestions[0].entity_name)
        self.assertEqual(options[1:], suggestions[0].values)

        self.assertIsNone(mongo_client.pdf_metadata_extraction.labeled_data.find_one({}))
