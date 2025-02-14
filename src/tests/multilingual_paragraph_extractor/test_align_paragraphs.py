from unittest import TestCase

from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures
from multilingual_paragraph_extractor.use_cases.MultilingualParagraphAlignerUseCase import (
    MultilingualParagraphAlignerUseCase,
)
from multilingual_paragraph_extractor.domain.ParagraphsFromLanguage import ParagraphsFromLanguage
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier


class TestAlignParagraphs(TestCase):
    extraction_identifier = ExtractionIdentifier(extraction_name="paragraph_extraction")

    def test_align_paragraphs_no_languages(self):
        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        paragraphs_from_languages = []
        multilingual_paragraph_extractor.align_languages(paragraphs_from_languages)

        self.assertEqual(0, len(paragraphs_from_languages))

    def test_align_paragraphs_only_one_language(self):
        pdf_data_paragraphs = ParagraphFeatures.from_texts(texts=["English text", "English text too"])
        language_paragraph = ParagraphsFromLanguage(language="en", paragraphs=pdf_data_paragraphs, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        paragraphs_from_languages = [language_paragraph]
        multilingual_paragraph_extractor.align_languages(paragraphs_from_languages)

        self.assertEqual(1, len(paragraphs_from_languages))
        self.assertEqual(2, len(paragraphs_from_languages[0]._aligned_paragraphs))

        self.assertEqual("en", paragraphs_from_languages[0].language)
        self.assertEqual("English text", paragraphs_from_languages[0]._aligned_paragraphs[0].text_cleaned)
        self.assertEqual("English text too", paragraphs_from_languages[0]._aligned_paragraphs[1].text_cleaned)

    def test_align_paragraphs_when_no_main_language(self):
        pdf_data_paragraphs_1 = ParagraphFeatures.from_texts(texts=["English text"])
        language_paragraph_1 = ParagraphsFromLanguage(
            language="en", paragraphs=pdf_data_paragraphs_1, is_main_language=False
        )

        pdf_data_paragraphs_2 = ParagraphFeatures.from_texts(texts=["French text"])
        language_paragraph_2 = ParagraphsFromLanguage(
            language="fr", paragraphs=pdf_data_paragraphs_2, is_main_language=False
        )

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        paragraphs_from_languages = [language_paragraph_1, language_paragraph_2]
        multilingual_paragraph_extractor.align_languages(paragraphs_from_languages)

        self.assertEqual(2, len(paragraphs_from_languages))
        self.assertEqual(1, len(paragraphs_from_languages[0]._aligned_paragraphs))
        self.assertEqual(1, len(paragraphs_from_languages[1]._aligned_paragraphs))

        self.assertEqual("en", paragraphs_from_languages[0].language)
        self.assertEqual("fr", paragraphs_from_languages[1].language)

        self.assertEqual("English text", paragraphs_from_languages[0]._aligned_paragraphs[0].text_cleaned)
        self.assertEqual("French text", paragraphs_from_languages[1]._aligned_paragraphs[0].text_cleaned)

    @staticmethod
    def get_paragraphs(language: str):
        paragraphs = ParagraphFeatures.from_texts(texts=[f"a 0. {language}", f"b 1: {language}", f"c 2! {language}"])
        return paragraphs

    def test_align_paragraphs_when_missing_paragraph_at_the_end(self):
        language_paragraph_1 = ParagraphsFromLanguage(
            language="en", paragraphs=self.get_paragraphs("en"), is_main_language=True
        )
        tr_paragraphs_missing_end = self.get_paragraphs("tr")[0:2]
        language_paragraph_2 = ParagraphsFromLanguage(
            language="tr", paragraphs=tr_paragraphs_missing_end, is_main_language=False
        )

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        paragraphs_from_languages = [language_paragraph_1, language_paragraph_2]
        multilingual_paragraph_extractor.align_languages(paragraphs_from_languages)

        self.assertEqual(2, len(paragraphs_from_languages))

        self.assertEqual(3, len(paragraphs_from_languages[0]._aligned_paragraphs))
        self.assertEqual(3, len(paragraphs_from_languages[1]._aligned_paragraphs))

        self.assertEqual("a 0. en", paragraphs_from_languages[0]._aligned_paragraphs[0].text_cleaned)
        self.assertEqual("a 0. tr", paragraphs_from_languages[1]._aligned_paragraphs[0].text_cleaned)

        self.assertEqual("b 1: en", paragraphs_from_languages[0]._aligned_paragraphs[1].text_cleaned)
        self.assertEqual("b 1: tr", paragraphs_from_languages[1]._aligned_paragraphs[1].text_cleaned)

        self.assertEqual("c 2! en", paragraphs_from_languages[0]._aligned_paragraphs[2].text_cleaned)
        self.assertEqual("", paragraphs_from_languages[1]._aligned_paragraphs[2].text_cleaned)

    def test_align_paragraphs_when_missing_middle_paragraph(self):
        language_paragraph_1 = ParagraphsFromLanguage(
            language="en", paragraphs=self.get_paragraphs("en"), is_main_language=True
        )
        tr_paragraphs_missing_middle = [self.get_paragraphs("tr")[0], self.get_paragraphs("tr")[2]]
        language_paragraph_2 = ParagraphsFromLanguage(
            language="tr", paragraphs=tr_paragraphs_missing_middle, is_main_language=False
        )

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        paragraphs_from_languages = [language_paragraph_1, language_paragraph_2]
        multilingual_paragraph_extractor.align_languages(paragraphs_from_languages)

        self.assertEqual(2, len(paragraphs_from_languages))

        self.assertEqual(3, len(paragraphs_from_languages[0]._aligned_paragraphs))
        self.assertEqual(3, len(paragraphs_from_languages[1]._aligned_paragraphs))

        self.assertEqual("a 0. en", paragraphs_from_languages[0]._aligned_paragraphs[0].text_cleaned)
        self.assertEqual("a 0. tr", paragraphs_from_languages[1]._aligned_paragraphs[0].text_cleaned)

        self.assertEqual("b 1: en", paragraphs_from_languages[0]._aligned_paragraphs[1].text_cleaned)
        self.assertEqual("", paragraphs_from_languages[1]._aligned_paragraphs[1].text_cleaned)

        self.assertEqual("c 2! en", paragraphs_from_languages[0]._aligned_paragraphs[2].text_cleaned)
        self.assertEqual("c 2! tr", paragraphs_from_languages[1]._aligned_paragraphs[2].text_cleaned)

    def test_align_paragraphs_when_missing_at_beginning(self):
        language_paragraph_1 = ParagraphsFromLanguage(
            language="en", paragraphs=self.get_paragraphs("en"), is_main_language=True
        )
        tr_paragraphs_missing_beginning = [self.get_paragraphs("tr")[1], self.get_paragraphs("tr")[2]]
        language_paragraph_2 = ParagraphsFromLanguage(
            language="tr", paragraphs=tr_paragraphs_missing_beginning, is_main_language=False
        )

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        paragraphs_from_languages = [language_paragraph_1, language_paragraph_2]
        multilingual_paragraph_extractor.align_languages(paragraphs_from_languages)

        self.assertEqual(2, len(paragraphs_from_languages))

        self.assertEqual(3, len(paragraphs_from_languages[0]._aligned_paragraphs))
        self.assertEqual(3, len(paragraphs_from_languages[1]._aligned_paragraphs))

        self.assertEqual("a 0. en", paragraphs_from_languages[0]._aligned_paragraphs[0].text_cleaned)
        self.assertEqual("", paragraphs_from_languages[1]._aligned_paragraphs[0].text_cleaned)

        self.assertEqual("b 1: en", paragraphs_from_languages[0]._aligned_paragraphs[1].text_cleaned)
        self.assertEqual("b 1: tr", paragraphs_from_languages[1]._aligned_paragraphs[1].text_cleaned)

        self.assertEqual("c 2! en", paragraphs_from_languages[0]._aligned_paragraphs[2].text_cleaned)
        self.assertEqual("c 2! tr", paragraphs_from_languages[1]._aligned_paragraphs[2].text_cleaned)

    def test_align_paragraphs_when_two_paragraphs_corresponds_to_one(self):
        language_paragraph_1 = ParagraphsFromLanguage(
            language="en", paragraphs=self.get_paragraphs("en"), is_main_language=True
        )
        paragraphs = self.get_paragraphs("tr")
        paragraphs_issue = [paragraphs[0], paragraphs[1].merge(paragraphs[2])]
        language_paragraph_2 = ParagraphsFromLanguage(language="tr", paragraphs=paragraphs_issue, is_main_language=False)

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        paragraphs_from_languages = [language_paragraph_1, language_paragraph_2]
        multilingual_paragraph_extractor.align_languages(paragraphs_from_languages)

        self.assertEqual(2, len(paragraphs_from_languages))

        self.assertEqual(3, len(paragraphs_from_languages[0]._aligned_paragraphs))
        self.assertEqual(3, len(paragraphs_from_languages[1]._aligned_paragraphs))

        self.assertEqual("a 0. en", paragraphs_from_languages[0]._aligned_paragraphs[0].text_cleaned)
        self.assertEqual("a 0. tr", paragraphs_from_languages[1]._aligned_paragraphs[0].text_cleaned)

        self.assertEqual("b 1: en", paragraphs_from_languages[0]._aligned_paragraphs[1].text_cleaned)
        self.assertEqual("b 1: tr", paragraphs_from_languages[1]._aligned_paragraphs[1].text_cleaned)

        self.assertEqual("c 2! en", paragraphs_from_languages[0]._aligned_paragraphs[2].text_cleaned)
        self.assertEqual("c 2! tr", paragraphs_from_languages[1]._aligned_paragraphs[2].text_cleaned)

    def test_align_paragraphs_when_one_paragraph_corresponds_to_two(self):
        paragraphs = self.get_paragraphs("en")
        paragraphs_issue = [paragraphs[0], paragraphs[1].merge(paragraphs[2])]
        language_paragraph_1 = ParagraphsFromLanguage(language="en", paragraphs=paragraphs_issue, is_main_language=True)

        language_paragraph_2 = ParagraphsFromLanguage(
            language="tr", paragraphs=self.get_paragraphs("tr"), is_main_language=False
        )

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        paragraphs_from_languages = [language_paragraph_1, language_paragraph_2]
        multilingual_paragraph_extractor.align_languages(paragraphs_from_languages)

        self.assertEqual(2, len(paragraphs_from_languages))

        self.assertEqual(3, len(paragraphs_from_languages[0]._aligned_paragraphs))
        self.assertEqual(3, len(paragraphs_from_languages[1]._aligned_paragraphs))

        self.assertEqual("a 0. en", paragraphs_from_languages[0]._aligned_paragraphs[0].text_cleaned)
        self.assertEqual("a 0. tr", paragraphs_from_languages[1]._aligned_paragraphs[0].text_cleaned)

        self.assertEqual("b 1: en", paragraphs_from_languages[0]._aligned_paragraphs[1].text_cleaned)
        self.assertEqual("b 1: tr", paragraphs_from_languages[1]._aligned_paragraphs[1].text_cleaned)

        self.assertEqual("c 2! en", paragraphs_from_languages[0]._aligned_paragraphs[2].text_cleaned)
        self.assertEqual("c 2! tr", paragraphs_from_languages[1]._aligned_paragraphs[2].text_cleaned)

    def test_two_different_pdfs(self):
        pdf_data_paragraphs_1 = ParagraphFeatures.from_texts(texts=["English text", "English text too"])
        language_paragraph_1 = ParagraphsFromLanguage(language="en", paragraphs=pdf_data_paragraphs_1, is_main_language=True)

        pdf_data_paragraphs_2 = ParagraphFeatures.from_texts(texts=["French text", "1. 34 ..;;;/// " * 4])
        language_paragraph_2 = ParagraphsFromLanguage(
            language="fr", paragraphs=pdf_data_paragraphs_2, is_main_language=False
        )

        multilingual_paragraph_extractor = MultilingualParagraphAlignerUseCase(
            extractor_identifier=self.extraction_identifier
        )
        paragraphs_from_languages = [language_paragraph_1, language_paragraph_2]
        multilingual_paragraph_extractor.align_languages(paragraphs_from_languages)

        self.assertEqual(2, len(paragraphs_from_languages))

        self.assertEqual(2, len(paragraphs_from_languages[0]._aligned_paragraphs))
        self.assertEqual(2, len(paragraphs_from_languages[1]._aligned_paragraphs))

        self.assertEqual("English text", paragraphs_from_languages[0]._aligned_paragraphs[0].original_text)
        self.assertEqual("English text too", paragraphs_from_languages[0]._aligned_paragraphs[1].original_text)

        self.assertEqual("", paragraphs_from_languages[1]._aligned_paragraphs[0].original_text)
        self.assertEqual("", paragraphs_from_languages[1]._aligned_paragraphs[1].original_text)
