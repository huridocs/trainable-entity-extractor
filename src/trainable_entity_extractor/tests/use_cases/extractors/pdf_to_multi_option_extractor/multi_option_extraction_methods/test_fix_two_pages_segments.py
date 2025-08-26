from unittest import TestCase

from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType

from trainable_entity_extractor.domain.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FastSegmentSelectorFuzzy95 import (
    FastSegmentSelectorFuzzy95,
)


class TestFixTwoPagesSegments(TestCase):
    def test_fix_two_pages_segments(self):
        seg1 = PdfDataSegment.from_values(
            page_number=1,
            bounding_box=Rectangle.from_coordinates(0, 0, 10, 10),
            text_content="First segment text",
            segment_type=TokenType.TEXT,
        )
        seg2 = PdfDataSegment.from_values(
            page_number=1,
            bounding_box=Rectangle.from_coordinates(0, 20, 10, 30),
            text_content="Second segment,",
            segment_type=TokenType.TEXT,
        )
        footer = PdfDataSegment.from_values(
            page_number=1,
            bounding_box=Rectangle.from_coordinates(0, 40, 10, 50),
            text_content="Footer info",
            segment_type=TokenType.PAGE_FOOTER,
        )
        header = PdfDataSegment.from_values(
            page_number=2,
            bounding_box=Rectangle.from_coordinates(0, 0, 10, 10),
            text_content="Header info",
            segment_type=TokenType.PAGE_HEADER,
        )
        seg3 = PdfDataSegment.from_values(
            page_number=2,
            bounding_box=Rectangle.from_coordinates(0, 20, 10, 30),
            text_content="Third segment first",
            segment_type=TokenType.TEXT,
        )
        seg4 = PdfDataSegment.from_values(
            page_number=2,
            bounding_box=Rectangle.from_coordinates(0, 40, 10, 50),
            text_content="Fourth segment",
            segment_type=TokenType.TEXT,
        )

        pdf_data = PdfData(file_name="dummy.pdf")
        pdf_data.pdf_data_segments = [seg1, seg2, footer, header, seg3, seg4]
        sample = TrainingSample(pdf_data=pdf_data)

        method = FastSegmentSelectorFuzzy95()
        fixed_segments = method.fix_two_pages_segments(sample)

        merged_texts = [s.text_content for s in fixed_segments]

        self.assertEqual(len(fixed_segments), 5, f"Unexpected segments count: {merged_texts}")
        self.assertIn("Second segment, Third segment first", merged_texts)
        self.assertNotIn("Third segment first", [t for t in merged_texts if t != "Second segment, Third segment first"])
        self.assertIn("Footer info", merged_texts)
        self.assertIn("Header info", merged_texts)
