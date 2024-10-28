import os
import shutil
from os.path import exists, join
from time import time

from unittest import TestCase

import mongomock

from trainable_entity_extractor.config import DATA_PATH, APP_PATH
from trainable_entity_extractor.data.ExtractionTask import ExtractionTask
from trainable_entity_extractor.data.Params import Params
from trainable_entity_extractor.Extractor import TrainableEntityExtractor


class TestRemoveModels(TestCase):
    test_xml_path = f"{APP_PATH}/tenant_test/extraction_id/xml_to_train/test.xml"
    model_path = f"{APP_PATH}/tenant_test/extraction_id/segment_predictor_model/model.model"

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_remove_models(self):
        extraction_id = "extraction_id"
        two_days_ago = (time() - (2 * 24 * 3600) - 1, time() - (2 * 24 * 3600) - 1)

        tenant_extractor_to_remove = "to_remove"

        os.makedirs(join(DATA_PATH, "cache", "folder_to_keep"), exist_ok=True)
        os.utime(join(DATA_PATH, "cache", "folder_to_keep"), two_days_ago)

        shutil.rmtree(join(DATA_PATH, tenant_extractor_to_remove), ignore_errors=True)
        shutil.copytree(join(APP_PATH, "tenant_test"), join(DATA_PATH, tenant_extractor_to_remove))
        os.utime(join(DATA_PATH, tenant_extractor_to_remove, extraction_id), two_days_ago)

        tenant_extractor_to_keep = "to_keep"

        shutil.rmtree(join(DATA_PATH, tenant_extractor_to_keep), ignore_errors=True)
        shutil.copytree(join(APP_PATH, "tenant_test"), join(DATA_PATH, tenant_extractor_to_keep))
        os.utime(join(DATA_PATH, tenant_extractor_to_keep, extraction_id), two_days_ago)

        task = ExtractionTask(
            tenant=tenant_extractor_to_keep,
            task=TrainableEntityExtractor.SUGGESTIONS_TASK_NAME,
            params=Params(id=extraction_id),
        )

        task_calculated, error = TrainableEntityExtractor.calculate_task(task)

        self.assertFalse(task_calculated)
        self.assertTrue(exists(join(DATA_PATH, "cache", "folder_to_keep")))
        self.assertTrue(exists(join(DATA_PATH, tenant_extractor_to_keep, extraction_id)))
        self.assertTrue(exists(join(DATA_PATH, tenant_extractor_to_remove)))
        self.assertFalse(exists(join(DATA_PATH, tenant_extractor_to_remove, extraction_id)))

        shutil.rmtree(join(DATA_PATH, tenant_extractor_to_remove), ignore_errors=True)
        shutil.rmtree(join(DATA_PATH, tenant_extractor_to_keep), ignore_errors=True)

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_do_nothing_set_time_not_existent_extractor(self):
        extraction_id = "not_existent_extractor"

        tenant_extractor_to_keep = "to_keep"

        shutil.rmtree(join(DATA_PATH, tenant_extractor_to_keep), ignore_errors=True)
        shutil.copytree(join(APP_PATH, "tenant_test"), join(DATA_PATH, tenant_extractor_to_keep))

        task = ExtractionTask(
            tenant=tenant_extractor_to_keep,
            task=TrainableEntityExtractor.SUGGESTIONS_TASK_NAME,
            params=Params(id=extraction_id),
        )

        task_calculated, error = TrainableEntityExtractor.calculate_task(task)

        self.assertFalse(task_calculated)

        shutil.rmtree(join(DATA_PATH, tenant_extractor_to_keep), ignore_errors=True)
