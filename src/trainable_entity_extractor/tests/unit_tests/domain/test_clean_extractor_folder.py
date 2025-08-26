import tempfile
from pathlib import Path as _Path
from unittest import TestCase

from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier


class TestCleanExtractorFolder(TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_path = _Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _create_extraction_identifier(self, run_name="run", extraction_name="extract"):
        return ExtractionIdentifier(run_name=run_name, extraction_name=extraction_name, output_path=self.base_path)

    def _make_dir(self, ei: ExtractionIdentifier, name: str):
        d = _Path(ei.get_path()) / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _make_file(self, ei: ExtractionIdentifier, name: str):
        f = _Path(ei.get_path()) / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("dummy")
        return f

    def test_nonexistent_path_no_error(self):
        ei = self._create_extraction_identifier()
        # Path not created on purpose
        ei.clean_extractor_folder()  # Should not raise
        self.assertFalse(_Path(ei.get_path()).exists())

    def test_deletes_keyword_directories_except_method_used(self):
        ei = self._create_extraction_identifier()
        # Prepare folders
        method_used_dir = "setfit_best_model"
        other_setfit = "setfit_old"
        t5_dir = "t5_temp"
        bert_dir = "bert_run"
        misc_dir = "misc"
        for name in [method_used_dir, other_setfit, t5_dir, bert_dir, misc_dir]:
            self._make_dir(ei, name)
        # File that includes keyword should remain (files skipped)
        self._make_file(ei, "bert_notes.txt")
        # Save method used so that matching folder is preserved
        ei.save_method_used(method_used_dir)

        ei.clean_extractor_folder()

        base = _Path(ei.get_path())
        existing_dirs = {p.name for p in base.iterdir() if p.is_dir()}
        # method_used_dir and non keyword misc should remain
        self.assertIn(method_used_dir, existing_dirs)
        self.assertIn(misc_dir, existing_dirs)
        # keyword dirs (other than method_used) removed
        self.assertNotIn(other_setfit, existing_dirs)
        self.assertNotIn(t5_dir, existing_dirs)
        self.assertNotIn(bert_dir, existing_dirs)
        # File still present
        self.assertTrue((base / "bert_notes.txt").is_file())

    def test_without_method_used_deletes_keyword_directories(self):
        ei = self._create_extraction_identifier()
        # Create directories without setting method_used.json
        dirs = ["setfit_model", "t5_en", "bert_base", "keepme"]
        for d in dirs:
            self._make_dir(ei, d)

        ei.clean_extractor_folder()

        base = _Path(ei.get_path())
        existing_dirs = {p.name for p in base.iterdir() if p.is_dir()}
        self.assertEqual(existing_dirs, {"keepme"})

    def test_method_used_case_insensitive_preserved(self):
        ei = self._create_extraction_identifier()
        # Folder created in lowercase
        preserved = "t5model"
        deleted = "t5_old"
        self._make_dir(ei, preserved)
        self._make_dir(ei, deleted)
        ei.save_method_used("T5Model")  # Different case

        ei.clean_extractor_folder()

        base = _Path(ei.get_path())
        existing_dirs = {p.name for p in base.iterdir() if p.is_dir()}
        self.assertIn(preserved, existing_dirs)
        self.assertNotIn(deleted, existing_dirs)
