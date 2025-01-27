from unittest import TestCase

from trainable_entity_extractor.data.PdfDataSegment import PdfDataSegment


class TestAreSameParagraph(TestCase):
    def test_are_same_paragraph(self):
        segment = PdfDataSegment.from_text("text")
        other_segment = PdfDataSegment.from_text("text")
        self.assertTrue(segment.are_similar(other_segment))

    def test_are_same_paragraph_same_numbers(self):
        segment = PdfDataSegment.from_text("text 10")
        other_segment = PdfDataSegment.from_text("10 text The Working Group on the Universal Periodic Review")
        self.assertTrue(segment.are_similar(other_segment))

    def test_are_same_paragraph_same_non_alphanumeric(self):
        segment = PdfDataSegment.from_text("text (///..,,;;?!)")
        other_segment = PdfDataSegment.from_text("(///) text The Working Group on the Universal Periodic Review")
        self.assertTrue(segment.are_similar(other_segment))

    def test_are_same_paragraph_with_similar_number_of_words(self):
        segment = PdfDataSegment.from_text("1 January 2021")
        other_segment = PdfDataSegment.from_text("first of January 2021")
        self.assertTrue(segment.are_similar(other_segment))

    def test_are_different_with_different_lengths(self):
        segment = PdfDataSegment.from_text("1 January 2021")
        other_segment = PdfDataSegment.from_text("first of January 2021, first of January 2021")
        self.assertFalse(segment.are_similar(other_segment))
