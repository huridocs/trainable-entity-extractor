from multilingual_paragraph_extractor.domain.AlignmentScore import AlignmentScore
from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures
from multilingual_paragraph_extractor.domain.ParagraphMatchScore import ParagraphMatchScore
from multilingual_paragraph_extractor.domain.ParagraphsFromLanguage import ParagraphsFromLanguage
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier


class MultilingualParagraphAlignerUseCase:
    def __init__(self, extractor_identifier: ExtractionIdentifier):
        self.extractor_identifier = extractor_identifier

    def align_languages(self, paragraphs_from_languages: list[ParagraphsFromLanguage]):
        if not paragraphs_from_languages:
            return []

        for paragraphs_from_language in paragraphs_from_languages:
            paragraphs_from_language.remove_no_text_types()
            paragraphs_from_language.merge_paragraphs_spanning_two_pages()

        main_language, other_languages = self.get_main_and_other_languages(paragraphs_from_languages)

        for other_language_paragraphs in other_languages:
            other_language_paragraphs.align(main_paragraphs=main_language.paragraphs)

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

    @staticmethod
    def align_language(
        main_language: ParagraphsFromLanguage, other_language: ParagraphsFromLanguage, alignment_scores: list[AlignmentScore]
    ):
        other_language_aligned_paragraphs = MultilingualParagraphAlignerUseCase.get_alignment_by_scores(
            alignment_scores, main_language
        )

        for idx, other_segment in enumerate(other_language.paragraphs):
            if other_segment in other_language_aligned_paragraphs:
                continue

            to_merge = other_language.paragraphs[idx - 1] if idx > 0 else None
            alignment_score = [score for score in alignment_scores if score.other_paragraph == other_segment][0]
            if MultilingualParagraphAlignerUseCase.should_merge_paragraphs(alignment_score, to_merge):
                pass

            to_merge = other_language.paragraphs[idx + 1] if idx + 1 < len(other_language.paragraphs) else None
            alignment_score = [score for score in alignment_scores if score.other_paragraph == other_segment][0]
            if MultilingualParagraphAlignerUseCase.should_merge_paragraphs(alignment_score, to_merge):
                pass

        other_language.paragraphs = other_language_aligned_paragraphs

    @staticmethod
    def get_alignment_by_scores(alignment_scores, main_language):
        other_language_aligned_paragraphs: list[ParagraphFeatures] = []
        alignment_dict = {score.main_paragraph: score.other_paragraph for score in alignment_scores}
        for segment in main_language.paragraphs:
            other_language_aligned_paragraphs.append(alignment_dict.get(segment, ParagraphFeatures.get_empty()))
        return other_language_aligned_paragraphs
