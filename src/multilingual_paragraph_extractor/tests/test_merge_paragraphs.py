from unittest import TestCase

from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures
from multilingual_paragraph_extractor.use_cases.MultilingualParagraphAlignerUseCase import (
    MultilingualParagraphAlignerUseCase,
)
from multilingual_paragraph_extractor.domain.SegmentsFromLanguage import ParagraphsFromLanguage
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier


class TestMergeParagraphs(TestCase):
    extraction_identifier = ExtractionIdentifier(extraction_name="paragraph_extraction")

    def test_merge_paragraphs_no_languages(self):
        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        segments_from_languages = []
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(0, len(segments_from_languages))

    def test_merge_paragraphs_only_one_language(self):
        pdf_data_segments = ParagraphFeatures.from_texts(texts=["English text", "English text too"])
        language_segment = ParagraphsFromLanguage(language="en", segments=pdf_data_segments, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        segments_from_languages = [language_segment]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(1, len(segments_from_languages))
        self.assertEqual(2, len(segments_from_languages[0].paragraphs))

        self.assertEqual("en", segments_from_languages[0].language)
        self.assertEqual("English text", segments_from_languages[0].paragraphs[0].text_content)
        self.assertEqual("English text too", segments_from_languages[0].paragraphs[1].text_content)

    def test_merge_paragraphs_when_no_main_language(self):
        pdf_data_segments_1 = ParagraphFeatures.from_texts(texts=["English text"])
        language_segment_1 = ParagraphsFromLanguage(language="en", segments=pdf_data_segments_1, is_main_language=False)

        pdf_data_segments_2 = ParagraphFeatures.from_texts(texts=["French text"])
        language_segment_2 = ParagraphsFromLanguage(language="fr", segments=pdf_data_segments_2, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        segments_from_languages = [language_segment_1, language_segment_2]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(2, len(segments_from_languages))
        self.assertEqual(1, len(segments_from_languages[0].paragraphs))
        self.assertEqual(1, len(segments_from_languages[1].paragraphs))

        self.assertEqual("en", segments_from_languages[0].language)
        self.assertEqual("fr", segments_from_languages[1].language)

        self.assertEqual("English text", segments_from_languages[0].paragraphs[0].text_content)
        self.assertEqual("French text", segments_from_languages[1].paragraphs[0].text_content)

    def get_segments(self, language: str):
        segments = ParagraphFeatures.from_texts(texts=[f"Text 0. {language}", f"Text 1. {language}", f"Text 2. {language}"])
        return segments

    def test_merge_paragraphs_when_missing_segment_at_the_end(self):
        language_segment_1 = ParagraphsFromLanguage(language="en", segments=self.get_segments("en"), is_main_language=True)
        tr_segments_missing_end = self.get_segments("tr")[0:2]
        language_segment_2 = ParagraphsFromLanguage(language="tr", segments=tr_segments_missing_end, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        segments_from_languages = [language_segment_1, language_segment_2]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(2, len(segments_from_languages))

        self.assertEqual(3, len(segments_from_languages[0].paragraphs))
        self.assertEqual(3, len(segments_from_languages[1].paragraphs))

        self.assertEqual("Text 0. en", segments_from_languages[0].paragraphs[0].text_content)
        self.assertEqual("Text 0. tr", segments_from_languages[1].paragraphs[0].text_content)

        self.assertEqual("Text 1. en", segments_from_languages[0].paragraphs[1].text_content)
        self.assertEqual("Text 1. tr", segments_from_languages[1].paragraphs[1].text_content)

        self.assertEqual("Text 2. en", segments_from_languages[0].paragraphs[2].text_content)
        self.assertEqual("", segments_from_languages[1].paragraphs[2].text_content)

    def test_merge_paragraphs_when_missing_middle_segment(self):
        language_segment_1 = ParagraphsFromLanguage(language="en", segments=self.get_segments("en"), is_main_language=True)
        tr_segments_missing_middle = [self.get_segments("tr")[0], self.get_segments("tr")[2]]
        language_segment_2 = ParagraphsFromLanguage(
            language="tr", segments=tr_segments_missing_middle, is_main_language=False
        )

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        segments_from_languages = [language_segment_1, language_segment_2]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(2, len(segments_from_languages))

        self.assertEqual(3, len(segments_from_languages[0].paragraphs))
        self.assertEqual(3, len(segments_from_languages[1].paragraphs))

        self.assertEqual("Text 0. en", segments_from_languages[0].paragraphs[0].text_content)
        self.assertEqual("Text 0. tr", segments_from_languages[1].paragraphs[0].text_content)

        self.assertEqual("Text 1. en", segments_from_languages[0].paragraphs[1].text_content)
        self.assertEqual("", segments_from_languages[1].paragraphs[1].text_content)

        self.assertEqual("Text 2. en", segments_from_languages[0].paragraphs[2].text_content)
        self.assertEqual("Text 2. tr", segments_from_languages[1].paragraphs[2].text_content)

    def test_merge_paragraphs_when_missing_at_beginning(self):
        language_segment_1 = ParagraphsFromLanguage(language="en", segments=self.get_segments("en"), is_main_language=True)
        tr_segments_missing_middle = [self.get_segments("tr")[1], self.get_segments("tr")[2]]
        language_segment_2 = ParagraphsFromLanguage(
            language="tr", segments=tr_segments_missing_middle, is_main_language=False
        )

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        segments_from_languages = [language_segment_1, language_segment_2]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(2, len(segments_from_languages))

        self.assertEqual(3, len(segments_from_languages[0].paragraphs))
        self.assertEqual(3, len(segments_from_languages[1].paragraphs))

        self.assertEqual("Text 0. en", segments_from_languages[0].paragraphs[0].text_content)
        self.assertEqual("", segments_from_languages[1].paragraphs[0].text_content)

        self.assertEqual("Text 1. en", segments_from_languages[0].paragraphs[1].text_content)
        self.assertEqual("Text 1. tr", segments_from_languages[1].paragraphs[1].text_content)

        self.assertEqual("Text 2. en", segments_from_languages[0].paragraphs[2].text_content)
        self.assertEqual("Text 2. tr", segments_from_languages[1].paragraphs[2].text_content)

    def test_merge_paragraphs_when_two_segments_corresponds_to_one(self):
        language_segment_1 = ParagraphsFromLanguage(language="en", segments=self.get_segments("en"), is_main_language=True)
        segments = self.get_segments("tr")
        segmentation_issue = [segments[0], segments[1].merge(segments[2])]
        language_segment_2 = ParagraphsFromLanguage(language="tr", segments=segmentation_issue, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        segments_from_languages = [language_segment_1, language_segment_2]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(2, len(segments_from_languages))

        self.assertEqual(3, len(segments_from_languages[0].paragraphs))
        self.assertEqual(3, len(segments_from_languages[1].paragraphs))

        self.assertEqual("Text 0. en", segments_from_languages[0].paragraphs[0].text_content)
        self.assertEqual("Text 0. tr", segments_from_languages[1].paragraphs[0].text_content)

        self.assertEqual("Text 1. en", segments_from_languages[0].paragraphs[1].text_content)
        self.assertEqual("Text 1. tr Text 2. tr", segments_from_languages[1].paragraphs[1].text_content)

        self.assertEqual("Text 2. en", segments_from_languages[0].paragraphs[2].text_content)
        self.assertEqual("", segments_from_languages[1].paragraphs[2].text_content)

    def test_merge_paragraphs_when_one_segments_corresponds_to_two(self):
        segments = self.get_segments("en")
        segmentation_issue = [segments[0], segments[1].merge(segments[2])]
        language_segment_1 = ParagraphsFromLanguage(language="en", segments=segmentation_issue, is_main_language=True)

        language_segment_2 = ParagraphsFromLanguage(language="tr", segments=self.get_segments("tr"), is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        segments_from_languages = [language_segment_1, language_segment_2]
        multilingual_paragraph_extractor.align_languages(segments_from_languages)

        self.assertEqual(2, len(segments_from_languages))

        self.assertEqual(2, len(segments_from_languages[0].paragraphs))
        self.assertEqual(2, len(segments_from_languages[1].paragraphs))

        self.assertEqual("Text 0. en", segments_from_languages[0].paragraphs[0].text_content)
        self.assertEqual("Text 0. tr", segments_from_languages[1].paragraphs[0].text_content)

        self.assertEqual("Text 1. en Text 2. en", segments_from_languages[0].paragraphs[1].text_content)
        self.assertEqual("Text 1. tr Text 2. tr", segments_from_languages[1].paragraphs[1].text_content)
