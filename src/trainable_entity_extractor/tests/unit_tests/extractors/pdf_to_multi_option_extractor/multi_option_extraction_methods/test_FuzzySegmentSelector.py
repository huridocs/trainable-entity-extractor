from unittest import TestCase
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzySegmentSelector import (
    FuzzySegmentSelector,
)


class TestFuzzySegmentSelector(TestCase):

    def test_get_appearances_exact_match(self):
        options = ["apple", "banana", "orange"]

        segments = [
            PdfDataSegment.from_text("I like apple and banana"),
            PdfDataSegment.from_text("Fresh banana from the market"),
            PdfDataSegment.from_text("Orange juice is delicious"),
        ]

        appearances = FuzzySegmentSelector.get_appearances(segments, options)

        self.assertEqual(4, len(appearances))

        appearance_labels = [app.option_label for app in appearances]
        self.assertIn("apple", appearance_labels)
        self.assertIn("banana", appearance_labels)
        self.assertIn("orange", appearance_labels)
        self.assertEqual(2, appearance_labels.count("banana"))

    def test_get_appearances_fuzzy_match(self):
        options = ["apple", "banana"]

        segments = [
            PdfDataSegment.from_text("I love apples and oranges"),
            PdfDataSegment.from_text("Banana smoothie"),
        ]

        appearances = FuzzySegmentSelector.get_appearances(segments, options)

        self.assertGreaterEqual(len(appearances), 1)

        appearance_labels = [app.option_label for app in appearances]
        self.assertIn("apple", appearance_labels)
        self.assertIn("banana", appearance_labels)

    def test_get_appearances_no_match(self):
        options = ["apple", "banana"]

        segments = [
            PdfDataSegment.from_text("I love strawberries"),
        ]

        appearances = FuzzySegmentSelector.get_appearances(segments, options)

        self.assertEqual(0, len(appearances))

    def test_get_cleaned_options(self):
        options = [
            Option(id="1", label="Red Apple"),
            Option(id="2", label="Green Apple"),
            Option(id="3", label="Banana"),
        ]

        fuzzy_selector = FuzzySegmentSelector(None)
        cleaned = fuzzy_selector.get_cleaned_options(options)

        self.assertEqual(3, len(cleaned))
        self.assertIn("red", cleaned[0])
        self.assertIn("green", cleaned[1])
        self.assertIn("banana", cleaned[2])

    def test_remove_accents(self):
        text_with_accents = "café résumé"
        result = FuzzySegmentSelector.remove_accents(text_with_accents)

        self.assertEqual("cafe resume", result)
