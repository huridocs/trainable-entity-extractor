from unittest import TestCase

from multilingual_paragraph_extractor.MultilingualParagraphExtractor import MultilingualParagraphExtractor
from multilingual_paragraph_extractor.domain.SegmentsFromLanguage import SegmentsFromLanguage
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.PdfDataSegment import PdfDataSegment


class TestMergeParagraphsAdvanced(TestCase):
    extraction_identifier = ExtractionIdentifier(extraction_name="paragraph_extraction")

    def get_segments(self, language: str):
        segments = PdfDataSegment.from_texts(texts=[f"Text 0. {language}", f"Text 1. {language}", f"Text 2. {language}"])
        return segments

    def test_merge_paragraphs_three_languages(self):
        language_segment_1 = SegmentsFromLanguage(language="en", segments=self.get_segments("en"), is_main_language=True)
        language_segment_2 = SegmentsFromLanguage(language="fr", segments=self.get_segments("fr"), is_main_language=False)
        language_segment_3 = SegmentsFromLanguage(language="tr", segments=self.get_segments("tr"), is_main_language=False)

        segments_from_languages = [language_segment_1, language_segment_2, language_segment_3]
        multilingual_paragraph_extractor = MultilingualParagraphExtractor(extractor_identifier=self.extraction_identifier)
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(3, len(segments_from_languages))

        self.assertEqual(3, len(segments_from_languages[0].segments))
        self.assertEqual(3, len(segments_from_languages[1].segments))
        self.assertEqual(3, len(segments_from_languages[2].segments))

        self.assertEqual("Text 0. en", segments_from_languages[0].segments[0].text_content)
        self.assertEqual("Text 0. fr", segments_from_languages[1].segments[0].text_content)
        self.assertEqual("Text 0. tr", segments_from_languages[2].segments[0].text_content)

        self.assertEqual("Text 1. en", segments_from_languages[0].segments[1].text_content)
        self.assertEqual("Text 1. fr", segments_from_languages[1].segments[1].text_content)
        self.assertEqual("Text 1. tr", segments_from_languages[2].segments[1].text_content)

        self.assertEqual("Text 2. en", segments_from_languages[0].segments[2].text_content)
        self.assertEqual("Text 2. fr", segments_from_languages[1].segments[2].text_content)
        self.assertEqual("Text 2. tr", segments_from_languages[2].segments[2].text_content)

    def test_merge_paragraphs_when_missing_segment_at_the_end(self):
        language_segment_1 = SegmentsFromLanguage(language="en", segments=self.get_segments("en"), is_main_language=True)
        tr_segments_missing_end = self.get_segments("tr")[0:2]
        language_segment_2 = SegmentsFromLanguage(language="tr", segments=tr_segments_missing_end, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphExtractor(extractor_identifier=self.extraction_identifier)
        segments_from_languages = [language_segment_1, language_segment_2]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(2, len(segments_from_languages))

        self.assertEqual(3, len(segments_from_languages[0].segments))
        self.assertEqual(3, len(segments_from_languages[1].segments))

        self.assertEqual("Text 0. en", segments_from_languages[0].segments[0].text_content)
        self.assertEqual("Text 0. tr", segments_from_languages[1].segments[0].text_content)

        self.assertEqual("Text 1. en", segments_from_languages[0].segments[1].text_content)
        self.assertEqual("Text 1. tr", segments_from_languages[1].segments[1].text_content)

        self.assertEqual("Text 2. en", segments_from_languages[0].segments[2].text_content)
        self.assertEqual("", segments_from_languages[1].segments[2].text_content)

    def test_merge_paragraphs_when_missing_middle_segment(self):
        language_segment_1 = SegmentsFromLanguage(language="en", segments=self.get_segments("en"), is_main_language=True)
        tr_segments_missing_middle = [self.get_segments("tr")[0], self.get_segments("tr")[2]]
        language_segment_2 = SegmentsFromLanguage(language="tr", segments=tr_segments_missing_middle, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphExtractor(extractor_identifier=self.extraction_identifier)
        segments_from_languages = [language_segment_1, language_segment_2]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(2, len(segments_from_languages))

        self.assertEqual(3, len(segments_from_languages[0].segments))
        self.assertEqual(3, len(segments_from_languages[1].segments))

        self.assertEqual("Text 0. en", segments_from_languages[0].segments[0].text_content)
        self.assertEqual("Text 0. tr", segments_from_languages[1].segments[0].text_content)

        self.assertEqual("Text 1. en", segments_from_languages[0].segments[1].text_content)
        self.assertEqual("", segments_from_languages[1].segments[1].text_content)

        self.assertEqual("Text 2. en", segments_from_languages[0].segments[2].text_content)
        self.assertEqual("Text 2. tr", segments_from_languages[1].segments[2].text_content)

    def test_merge_paragraphs_when_missing_at_beginning(self):
        language_segment_1 = SegmentsFromLanguage(language="en", segments=self.get_segments("en"), is_main_language=True)
        tr_segments_missing_middle = [self.get_segments("tr")[1], self.get_segments("tr")[2]]
        language_segment_2 = SegmentsFromLanguage(language="tr", segments=tr_segments_missing_middle, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphExtractor(extractor_identifier=self.extraction_identifier)
        segments_from_languages = [language_segment_1, language_segment_2]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(2, len(segments_from_languages))

        self.assertEqual(3, len(segments_from_languages[0].segments))
        self.assertEqual(3, len(segments_from_languages[1].segments))

        self.assertEqual("Text 0. en", segments_from_languages[0].segments[0].text_content)
        self.assertEqual("", segments_from_languages[1].segments[0].text_content)

        self.assertEqual("Text 1. en", segments_from_languages[0].segments[1].text_content)
        self.assertEqual("Text 1. tr", segments_from_languages[1].segments[1].text_content)

        self.assertEqual("Text 2. en", segments_from_languages[0].segments[2].text_content)
        self.assertEqual("Text 2. tr", segments_from_languages[1].segments[2].text_content)

    def test_merge_paragraphs_when_two_segments_in_other_language(self):
        language_segment_1 = SegmentsFromLanguage(language="en", segments=self.get_segments("en"), is_main_language=True)
        segments = self.get_segments("tr")
        segmentation_issue = [segments[0], PdfDataSegment.from_list_to_merge(segments[1:])]
        language_segment_2 = SegmentsFromLanguage(language="tr", segments=segmentation_issue, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphExtractor(extractor_identifier=self.extraction_identifier)
        segments_from_languages = [language_segment_1, language_segment_2]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(2, len(segments_from_languages))

        self.assertEqual(3, len(segments_from_languages[0].segments))
        self.assertEqual(3, len(segments_from_languages[1].segments))

        self.assertEqual("Text 0. en", segments_from_languages[0].segments[0].text_content)
        self.assertEqual("", segments_from_languages[1].segments[0].text_content)

        self.assertEqual("Text 1. en", segments_from_languages[0].segments[1].text_content)
        self.assertEqual("Text 1. tr", segments_from_languages[1].segments[1].text_content)

        self.assertEqual("Text 2. en", segments_from_languages[0].segments[2].text_content)
        self.assertEqual("Text 2. tr", segments_from_languages[1].segments[2].text_content)
