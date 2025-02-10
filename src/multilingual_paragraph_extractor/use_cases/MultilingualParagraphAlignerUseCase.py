from multilingual_paragraph_extractor.domain.ParagraphsFromLanguage import ParagraphsFromLanguage
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier


class MultilingualParagraphAlignerUseCase:
    def __init__(self, extractor_identifier: ExtractionIdentifier):
        self.extractor_identifier = extractor_identifier

    def align_languages(self, paragraphs_from_languages: list[ParagraphsFromLanguage]):
        if not paragraphs_from_languages:
            return []

        for paragraphs_from_language in paragraphs_from_languages:
            paragraphs_from_language.remove_no_text_paragraphs()
            paragraphs_from_language.remove_headers_and_footers()
            paragraphs_from_language.merge_paragraphs_spanning_two_pages()
            paragraphs_from_language.remove_no_text_types()

        main_language, other_languages = self.get_main_and_other_languages(paragraphs_from_languages)

        for other_language_paragraphs in other_languages:
            other_language_paragraphs.align(main_language)

    @staticmethod
    def get_main_and_other_languages(
        paragraphs_from_languages: list[ParagraphsFromLanguage],
    ) -> tuple[ParagraphsFromLanguage, list[ParagraphsFromLanguage]]:
        main_languages = [x for x in paragraphs_from_languages if x.is_main_language]
        if not main_languages:
            return paragraphs_from_languages[0], paragraphs_from_languages[1:]

        main_language = main_languages[0]
        other_languages = [x for x in paragraphs_from_languages if x != main_language]
        return main_language, other_languages
