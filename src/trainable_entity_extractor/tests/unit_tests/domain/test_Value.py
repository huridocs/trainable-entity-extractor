from unittest import TestCase
from trainable_entity_extractor.domain.Value import Value


class TestValue(TestCase):
    def test_exact_match_case_sensitive(self):
        value = Value(id="1", label="item 1", segment_text="This is item 1 in the text")
        self.assertIn("<b>item 1</b>", value.segment_text)
        self.assertTrue(value.segment_text.startswith("<p>"))
        self.assertTrue(value.segment_text.endswith("</p>"))

    def test_exact_match_case_insensitive(self):
        value = Value(id="1", label="Item 1", segment_text="This is ITEM 1 in the text")
        self.assertIn("<b>ITEM 1</b>", value.segment_text)

    def test_multiple_exact_matches(self):
        value = Value(id="1", label="test", segment_text="test this test again test")
        expected = "<p><b>test</b> this <b>test</b> again <b>test</b></p>"
        self.assertEqual(value.segment_text, expected)

    def test_fuzzy_match_with_typo(self):
        value = Value(id="1", label="item", segment_text="This is itme in the text")
        self.assertIn("<b>itm</b>", value.segment_text)

    def test_fuzzy_match_with_extra_characters(self):
        value = Value(id="1", label="item", segment_text="This is items in the text")
        self.assertIn("<b>item</b>", value.segment_text)

    def test_fuzzy_match_partial_word(self):
        value = Value(id="1", label="cat", segment_text="The cats are playing")
        self.assertIn("<b>cat</b>", value.segment_text)

    def test_no_match_below_threshold(self):
        value = Value(id="1", label="completely different", segment_text="This is item 1 in the text")
        self.assertEqual(value.segment_text, "<p>This is item 1 in the text</p>")
        self.assertNotIn("<b>", value.segment_text)

    def test_empty_label(self):
        value = Value(id="1", label="", segment_text="This is some text")
        self.assertEqual(value.segment_text, "<p>This is some text</p>")
        self.assertNotIn("<b>", value.segment_text)

    def test_empty_segment_text(self):
        value = Value(id="1", label="item", segment_text="")
        self.assertEqual(value.segment_text, "")

    def test_whitespace_only_label(self):
        value = Value(id="1", label="   ", segment_text="This is some text")
        self.assertEqual(value.segment_text, "<p>This is some text</p>")
        self.assertNotIn("<b>", value.segment_text)

    def test_html_escaping_in_text(self):
        value = Value(id="1", label="test", segment_text="This has <test> & other content")
        self.assertIn("&lt;<b>test</b>&gt;", value.segment_text)
        self.assertIn("&amp;", value.segment_text)

    def test_html_escaping_with_match(self):
        value = Value(id="1", label="<test>", segment_text="This has <test> content")
        self.assertIn("<b>&lt;test&gt;</b>", value.segment_text)

    def test_existing_html_not_processed(self):
        original_text = "This has <b>bold</b> text"
        value = Value(id="1", label="test", segment_text=original_text)
        self.assertEqual(value.segment_text, original_text)

    def test_existing_br_tags_not_processed(self):
        original_text = "This has <br> tags"
        value = Value(id="1", label="test", segment_text=original_text)
        self.assertEqual(value.segment_text, original_text)

    def test_special_regex_characters_in_label(self):
        value = Value(id="1", label="test.*+", segment_text="This is test.*+ content")
        self.assertIn("<b>test.*+</b>", value.segment_text)

    def test_fuzzy_match_selects_best_score(self):
        value = Value(id="1", label="item", segment_text="This has itme and items in it")
        self.assertIn("<b>", value.segment_text)
        self.assertEqual(value.segment_text.count("<b>"), 1)

    def test_exact_match_preferred_over_fuzzy(self):
        value = Value(id="1", label="item", segment_text="This has item and itme in it")
        self.assertIn("<b>item</b>", value.segment_text)
        self.assertNotIn("<b>itme</b>", value.segment_text)

    def test_fuzzy_match_minimum_score_threshold(self):
        value = Value(id="1", label="hello", segment_text="This has help in it")
        score_help_hello = 80
        value2 = Value(id="2", label="programming", segment_text="This has coding in it")
        self.assertNotIn("<b>coding</b>", value2.segment_text)

    def test_unicode_characters(self):
        value = Value(id="1", label="café", segment_text="I like café and coffee")
        self.assertIn("<b>café</b>", value.segment_text)

    def test_fuzzy_match_with_punctuation(self):
        value = Value(id="1", label="item-1", segment_text="This is item_1 in the text")
        self.assertIn("<b>item_1</b>", value.segment_text)
