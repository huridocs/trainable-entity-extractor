import json
import shutil
from os.path import join
from unittest import TestCase

from trainable_entity_extractor.TrainableEntityExtractor import TrainableEntityExtractor
from trainable_entity_extractor.config import DATA_PATH, APP_PATH
from trainable_entity_extractor.data.ExtractionTask import ExtractionTask
from trainable_entity_extractor.data.Option import Option
from trainable_entity_extractor.data.Params import Params
from trainable_entity_extractor.data.SegmentBox import SegmentBox
from trainable_entity_extractor.data.Suggestion import Suggestion


class TestExtractorPdfToMultiOption(TestCase):
    def test_get_pdf_multi_option_suggestions(self):

        tenant = "tenant_to_be_removed"
        extraction_id = "extraction_id"

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)
        shutil.copytree(f"{APP_PATH}/tenant_test", f"{DATA_PATH}/{tenant}")

        labeled_data_json = {
            "id": extraction_id,
            "tenant": tenant,
            "xml_file_name": "test.xml",
            "language_iso": "en",
            "values": [{"id": "id15", "label": "15"}],
            "page_width": 612,
            "page_height": 792,
        }

        mongo_client.pdf_metadata_extraction.labeled_data.insert_one(labeled_data_json)

        options = [Option(id=f"id{n}", label=str(n)) for n in range(16)]

        TrainableEntityExtractor.calculate_task(
            ExtractionTask(
                tenant=tenant,
                task=TrainableEntityExtractor.CREATE_MODEL_TASK_NAME,
                params=Params(id=extraction_id, options=options, multi_value=False),
            )
        )

        to_predict_json = [
            {
                "xml_file_name": "test.xml",
                "id": extraction_id,
                "tenant": tenant,
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
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
        self.assertEqual("test.xml", suggestions[0].xml_file_name)
        self.assertEqual([Option(id="id15", label="15")], suggestions[0].values)

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_context_multi_option_suggestions(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "tenant_to_be_removed"
        extraction_id = "extraction_id"

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)
        shutil.copytree(f"{APP_PATH}/tenant_test", f"{DATA_PATH}/{tenant}")

        to_predict_json = [
            {
                "xml_file_name": "test.xml",
                "id": extraction_id,
                "tenant": tenant,
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
            }
        ]

        mongo_client.pdf_metadata_extraction.prediction_data.insert_many(to_predict_json)

        for i in range(7):
            labeled_data_json = {
                "id": extraction_id,
                "tenant": tenant,
                "xml_file_name": "test.xml",
                "language_iso": "en",
                "values": [{"id": "id15", "label": "15"}],
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
                "label_segments_boxes": [
                    json.loads(
                        SegmentBox(
                            left=397, top=91, width=10, height=9, page_width=612, page_height=792, page_number=1
                        ).model_dump_json()
                    )
                ],
            }

            mongo_client.pdf_metadata_extraction.labeled_data.insert_one(labeled_data_json)

        options = [Option(id=f"id{n}", label=str(n)) for n in range(16)]

        TrainableEntityExtractor.calculate_task(
            ExtractionTask(
                tenant=tenant,
                task=TrainableEntityExtractor.CREATE_MODEL_TASK_NAME,
                params=Params(id=extraction_id, options=options, multi_value=False),
            )
        )

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
        self.assertEqual("test.xml", suggestions[0].xml_file_name)
        self.assertEqual([Option(id="id15", label="15")], suggestions[0].values)

        self.assertIsNone(mongo_client.pdf_metadata_extraction.labeled_data.find_one({}))
