from unittest import TestCase
from trainable_entity_extractor.domain.FormatSegmentText import FormatSegmentText


class TestFormatSegmentText(TestCase):
    def test_multi_segment(self):
        formatter = FormatSegmentText(["Line 1. Same line", "Line 2"], "")
        result = formatter.format()
        self.assertEqual('<p class="ix_paragraph">Line 1. Same line</p><p class="ix_paragraph">Line 2</p>', result)

    def test_exact_match_case_sensitive(self):
        formatter = FormatSegmentText(["This is item 1 in the text"], "item 1")
        result = formatter.format()
        self.assertIn('<span class="ix_match">item 1</span>', result)
        self.assertTrue(result.startswith('<p class="ix_matching_paragraph">'))
        self.assertTrue(result.endswith("</p>"))

    def test_exact_match_case_insensitive(self):
        formatter = FormatSegmentText(["This is ITEM 1 in the text"], "Item 1")
        result = formatter.format()
        self.assertIn('<span class="ix_match">ITEM 1</span>', result)

    def test_multiple_exact_matches(self):
        formatter = FormatSegmentText(["test this test again test"], "test")
        result = formatter.format()
        expected = '<p class="ix_matching_paragraph"><span class="ix_match">test</span> this <span class="ix_match">test</span> again <span class="ix_match">test</span></p>'
        self.assertEqual(result, expected)

    def test_fuzzy_match_with_typo(self):
        formatter = FormatSegmentText(["This is itme in the text"], "item")
        result = formatter.format()
        self.assertIn('<span class="ix_match">itme</span>', result)

    def test_fuzzy_match_with_extra_characters(self):
        formatter = FormatSegmentText(["This is items in the text"], "item")
        result = formatter.format()
        self.assertIn('<span class="ix_match">item</span>', result)

    def test_fuzzy_match_partial_word(self):
        formatter = FormatSegmentText(["The cats are playing"], "cat")
        result = formatter.format()
        self.assertIn('<span class="ix_match">cat</span>', result)

    def test_no_match_below_threshold(self):
        formatter = FormatSegmentText(["This is item 1 in the text"], "completely different")
        result = formatter.format()
        self.assertEqual(result, '<p class="ix_paragraph">This is item 1 in the text</p>')
        self.assertNotIn('<span class="ix_match">', result)

    def test_empty_label(self):
        formatter = FormatSegmentText(["This is some text"], "")
        result = formatter.format()
        self.assertEqual(result, '<p class="ix_paragraph">This is some text</p>')
        self.assertNotIn('<span class="ix_match">', result)

    def test_empty_segment_text(self):
        formatter = FormatSegmentText([], "item")
        result = formatter.format()
        self.assertEqual(result, "")

    def test_whitespace_only_label(self):
        formatter = FormatSegmentText(["This is some text"], "   ")
        result = formatter.format()
        self.assertEqual(result, '<p class="ix_paragraph">This is some text</p>')
        self.assertNotIn('<span class="ix_match">', result)

    def test_html_escaping_in_text(self):
        formatter = FormatSegmentText(["This has <test> & other content"], "test")
        result = formatter.format()
        self.assertIn('&lt;<span class="ix_match">test</span>&gt;', result)
        self.assertIn("&amp;", result)

    def test_html_escaping_with_match(self):
        formatter = FormatSegmentText(["This has <test> content"], "<test>")
        result = formatter.format()
        self.assertIn('<span class="ix_match">&lt;test&gt;</span>', result)

    def test_special_regex_characters_in_label(self):
        formatter = FormatSegmentText(["This is test.*+ content"], "test.*+")
        result = formatter.format()
        self.assertIn('<span class="ix_match">test.*+</span>', result)

    def test_fuzzy_match_selects_best_score(self):
        formatter = FormatSegmentText(["This has itme and items in it"], "item")
        result = formatter.format()
        self.assertIn('<span class="ix_match">', result)
        self.assertEqual(result.count('<span class="ix_match">'), 1)

    def test_exact_match_preferred_over_fuzzy(self):
        formatter = FormatSegmentText(["This has item and itme in it"], "item")
        result = formatter.format()
        self.assertIn('<span class="ix_match">item</span>', result)
        self.assertNotIn('<span class="ix_match">itme</span>', result)

    def test_fuzzy_match_minimum_score_threshold(self):
        formatter = FormatSegmentText(["This has help in it"], "hello")
        result = formatter.format()
        formatter2 = FormatSegmentText(["This has coding in it"], "programming")
        result2 = formatter2.format()
        self.assertNotIn('<span class="ix_match">coding</span>', result2)

    def test_unicode_characters(self):
        formatter = FormatSegmentText(["I like café and coffee"], "café")
        result = formatter.format()
        self.assertIn('<span class="ix_match">café</span>', result)

    def test_fuzzy_match_with_punctuation(self):
        formatter = FormatSegmentText(["This is item_1 in the text"], "item-1")
        result = formatter.format()
        self.assertIn('<span class="ix_match">item_1</span>', result)

    def test_no_label_parameter(self):
        formatter = FormatSegmentText(["This is some text"])
        result = formatter.format()
        self.assertEqual(result, '<p class="ix_paragraph">This is some text</p>')
        self.assertNotIn('<span class="ix_match">', result)

    def test_format_with_none_label(self):
        formatter = FormatSegmentText(["This is some text"], None)
        result = formatter.format()
        self.assertEqual(result, '<p class="ix_paragraph">This is some text</p>')
        self.assertNotIn('<span class="ix_match">', result)

    def test_multiple_texts_no_label(self):
        formatter = FormatSegmentText(["First text", "Second text", "Third text"])
        result = formatter.format()
        self.assertEqual(result, '<p class="ix_paragraph">First text</p><p class="ix_paragraph">Second text</p><p class="ix_paragraph">Third text</p>')
        self.assertNotIn('<span class="ix_match">', result)

    def test_multiple_texts_with_label(self):
        formatter = FormatSegmentText(["First item text", "Second text", "Third item text"], "item")
        result = formatter.format()
        self.assertIn('<span class="ix_match">item</span>', result)
        self.assertTrue(result.startswith('<p class="ix_matching_paragraph">'))
        self.assertTrue(result.endswith("</p>"))

    def test_empty_strings_in_list(self):
        formatter = FormatSegmentText(["First text", "", "Third text"], "text")
        result = formatter.format()
        self.assertEqual('<p class="ix_matching_paragraph">First <span class="ix_match">text</span></p><p class="ix_adjacent_paragraph"></p><p class="ix_matching_paragraph">Third <span class="ix_match">text</span></p>', result)

    def test_formating_dates(self):
        formatter = FormatSegmentText(["United Nations6", "General 6 October 2010"], "2010/10/06")
        result = formatter.format()
        self.assertEqual('<p class="ix_adjacent_paragraph">United Nations6</p><p class="ix_matching_paragraph">General <span class="ix_match">6</span> <span class="ix_match">October</span> <span class="ix_match">2010</span></p>', result)

    def test_formating_other_date(self):
        formatter = FormatSegmentText(["United Nations1", "General 1 September 2005"], "2005/09/01")
        result = formatter.format()
        self.assertEqual('<p class="ix_adjacent_paragraph">United Nations1</p><p class="ix_matching_paragraph">General <span class="ix_match">1</span> <span class="ix_match">September</span> <span class="ix_match">2005</span></p>', result)

    def test_list_with_only_empty_strings(self):
        formatter = FormatSegmentText(["", "", ""], "item")
        result = formatter.format()
        self.assertEqual(result, "")

    def test_shorter_segment_text(self):
        texts = [
            "1",
            "2",
            "3",
            "Text appearance",
            "4",
            "5",
            "6",
            "7",
        ]
        formatter = FormatSegmentText(texts, "text")
        result = formatter.format()
        self.assertEqual('<p class="ix_adjacent_paragraph">3</p><p class="ix_matching_paragraph"><span class="ix_match">Text</span> appearance</p><p class="ix_adjacent_paragraph">4</p>', result)

    def test_shorter_segment_text_for_dates(self):
        texts = [
            "Header",
            "Meeting 6 October",
            "Report 2010 results",
            "Closing",
            "Tail",
        ]
        formatter = FormatSegmentText(texts, "2010/10/06")
        result = formatter.format()
        self.assertEqual(
            '<p class="ix_adjacent_paragraph">Header</p><p class="ix_matching_paragraph">Meeting <span class="ix_match">6</span> <span class="ix_match">October</span></p><p class="ix_matching_paragraph">Report <span class="ix_match">2010</span> results</p><p class="ix_adjacent_paragraph">Closing</p>',
            result,
        )
