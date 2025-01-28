from pdf_token_type_labels.TokenType import TokenType

from multilingual_paragraph_extractor.domain.AlignmentScore import AlignmentScore
from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures
from multilingual_paragraph_extractor.domain.ParagraphMatchScore import ParagraphMatchScore
from multilingual_paragraph_extractor.domain.SegmentsFromLanguage import SegmentsFromLanguage
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.PdfDataSegment import PdfDataSegment


class MultilingualParagraphAlignerUseCase:
    BLOCK_SIZE = 20
    THRESHOLD = 0.7

    def __init__(self, extractor_identifier: ExtractionIdentifier):
        self.extractor_identifier = extractor_identifier

    def align_languages(self, segments_from_languages: list[SegmentsFromLanguage]):
        if not segments_from_languages:
            return []

        segments_from_languages = [self.remove_no_text_types(x) for x in segments_from_languages]
        segments_from_languages = [self.merge_segments_spanning_two_pages(x) for x in segments_from_languages]

        main_language, other_languages = self.get_main_and_other_languages(segments_from_languages)

        for language_segments in other_languages:
            alignment_scores = self.get_alignment_scores(
                main_segments=main_language.segments, other_segments=language_segments.segments
            )
            self.align_language(main_language, language_segments, alignment_scores)

    @staticmethod
    def get_main_and_other_languages(
        segments_from_languages: list[SegmentsFromLanguage],
    ) -> tuple[SegmentsFromLanguage, list[SegmentsFromLanguage]]:
        main_languages = [x for x in segments_from_languages if x.is_main_language]
        if not main_languages:
            return segments_from_languages[0], segments_from_languages[1:]

        main_language = main_languages[0]
        other_languages = [x for x in segments_from_languages if x != main_language]
        return main_language, other_languages

    @staticmethod
    def remove_no_text_types(segments_from_language: SegmentsFromLanguage) -> SegmentsFromLanguage:
        text_content_types = [
            TokenType.FORMULA,
            TokenType.LIST_ITEM,
            TokenType.TITLE,
            TokenType.TEXT,
            TokenType.SECTION_HEADER,
        ]
        segments = [x for x in segments_from_language.segments if x.segment_type in text_content_types]
        segments_from_language.segments = segments
        return segments_from_language

    def merge_segments_spanning_two_pages(self, segments_from_language: SegmentsFromLanguage) -> SegmentsFromLanguage:
        fixed_segments = []
        segments = segments_from_language.segments
        index = 0

        while index < len(segments):
            segment = segments[index]
            if index + 1 < len(segments) and self.are_same_segment_from_same_language(segment, segments[index + 1]):
                merged_segment = PdfDataSegment.from_list_to_merge([segment, segments[index + 1]])
                fixed_segments.append(merged_segment)
                index += 2
            else:
                fixed_segments.append(segment)
                index += 1

        segments_from_language.segments = fixed_segments
        return segments_from_language

    @staticmethod
    def are_same_segment_from_same_language(segment: ParagraphFeatures, next_segment: ParagraphFeatures) -> bool:
        if segment.page_number == next_segment.page_number:
            return False

        if int(segment.page_number - next_segment.page_number) > 1:
            return False

        if segment.segment_type != next_segment.segment_type:
            return False

        if segment.text_content[-1] in [".", "!", "?", ";"]:
            return False

        return True

    def get_alignment_scores(
        self, main_segments: list[ParagraphFeatures], other_segments: list[ParagraphFeatures]
    ) -> list[AlignmentScore]:
        matches: list[AlignmentScore] = []
        unmatched2 = set(range(len(other_segments)))

        for idx1 in range(len(main_segments)):
            start_j = max(0, idx1 - self.BLOCK_SIZE)  # Look behind
            end_j = min(len(other_segments), idx1 + self.BLOCK_SIZE)  # Look ahead
            current_block2 = list(unmatched2 & set(range(start_j, end_j)))

            best_match = None
            best_score = self.THRESHOLD

            for idx2 in current_block2:
                match_score = ParagraphMatchScore.from_paragraphs_features(main_segments[idx1], other_segments[idx2])
                score = match_score.overall_score
                if score > best_score:
                    best_score = score
                    best_match = idx2
                    if score > 0.95:
                        break

            if best_match is not None:
                alignment_score = AlignmentScore(
                    main_paragraph=main_segments[idx1], other_paragraph=other_segments[best_match], score=best_score
                )
                matches.append(alignment_score)
                unmatched2.remove(best_match)

        return matches
