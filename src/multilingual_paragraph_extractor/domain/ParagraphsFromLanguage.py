from pdf_token_type_labels.TokenType import TokenType
from pydantic import BaseModel

from multilingual_paragraph_extractor.domain.AlignmentScore import AlignmentScore
from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures
from multilingual_paragraph_extractor.domain.ParagraphMatchScore import ParagraphMatchScore

BLOCK_SIZE = 20
THRESHOLD = [0.9, 0.86, 0.82, 0.78]


class ParagraphsFromLanguage(BaseModel):
    language: str
    paragraphs: list[ParagraphFeatures]
    is_main_language: bool
    _alignment_scores: dict[ParagraphFeatures, AlignmentScore] = dict()

    class Config:
        arbitrary_types_allowed = True

    def remove_no_text_types(self):
        text_content_types = [
            TokenType.LIST_ITEM,
            TokenType.TEXT,
        ]
        self.paragraphs = [x for x in self.paragraphs if x.paragraph_type in text_content_types]

    def remove_no_text_paragraphs(self):
        self.paragraphs = [x for x in self.paragraphs if x.text_cleaned]
        self.paragraphs = [x for x in self.paragraphs if any(char.isalnum() for char in x.text_cleaned)]

    def remove_headers_and_footers(self):
        types = [TokenType.FOOTNOTE, TokenType.PAGE_HEADER, TokenType.PAGE_FOOTER]
        self.paragraphs = [x for x in self.paragraphs if x.paragraph_type not in types]

    def merge_paragraphs_spanning_two_pages(self):
        fixed_paragraphs = []
        index = 0

        while index < len(self.paragraphs):
            paragraph = self.paragraphs[index]

            if index + 1 >= len(self.paragraphs):
                fixed_paragraphs.append(paragraph)
                index += 1
                continue

            if paragraph.is_similar(self.paragraphs[index + 1]):
                merged_segment = paragraph.merge(self.paragraphs[index + 1])
                fixed_paragraphs.append(merged_segment)
                index += 2
                continue

            fixed_paragraphs.append(paragraph)
            index += 1

        self.paragraphs = fixed_paragraphs

    def set_alignment_scores(self, main_paragraphs: list[ParagraphFeatures]):
        unmatched_1 = set(range(len(main_paragraphs)))
        unmatched_2 = set(range(len(self.paragraphs)))

        for threshold in THRESHOLD:
            for idx1 in list(unmatched_1):
                start_j = max(0, idx1 - BLOCK_SIZE)
                end_j = min(len(self.paragraphs), idx1 + BLOCK_SIZE)
                current_block2 = list(unmatched_2 & set(range(start_j, end_j)))

                best_match = None
                best_score = threshold

                for idx2 in current_block2:
                    match_score = ParagraphMatchScore.from_paragraphs_features(main_paragraphs[idx1], self.paragraphs[idx2])
                    score = match_score.overall_score
                    if score > best_score:
                        best_score = score
                        best_match = idx2
                        if score > 0.95:
                            break

                if best_match is not None:
                    alignment_score = AlignmentScore(
                        main_paragraph=main_paragraphs[idx1], other_paragraph=self.paragraphs[best_match], score=best_score
                    )
                    self._alignment_scores[main_paragraphs[idx1]] = alignment_score
                    unmatched_2.remove(best_match)
                    unmatched_1.remove(idx1)

    def align(self, main_language: "ParagraphsFromLanguage"):
        self.set_alignment_scores(main_language.paragraphs)
        aligned_paragraphs = self.get_aligned_paragraphs_from_scores(main_language)
        self.assign_missing_in_both_languages(main_language.paragraphs, aligned_paragraphs)
        self.assign_unmatch_paragraphs()
        self.paragraphs = aligned_paragraphs

    def get_aligned_paragraphs_from_scores(self, main_language):
        aligned_paragraphs: list[ParagraphFeatures] = []
        for paragraph in main_language.paragraphs:
            if paragraph in self._alignment_scores:
                paragraph_to_add = self._alignment_scores[paragraph].other_paragraph
            else:
                paragraph_to_add = ParagraphFeatures.get_empty()

            aligned_paragraphs.append(paragraph_to_add)
        return aligned_paragraphs

    def assign_missing_in_both_languages(
        self, main_paragraphs: list[ParagraphFeatures], aligned_paragraphs: list[ParagraphFeatures]
    ):
        for idx, main_paragraph in enumerate(main_paragraphs):
            if main_paragraph in self._alignment_scores:
                continue

            main_previous = main_paragraphs[idx - 1] if idx > 0 else None
            if main_previous not in self._alignment_scores:
                continue

            previous_index = self.paragraphs.index(self._alignment_scores[main_previous].other_paragraph)
            paragraph = self.paragraphs[previous_index + 1] if previous_index + 1 < len(self.paragraphs) else None
            if not paragraph or paragraph in aligned_paragraphs:
                continue

            alignment_score = ParagraphMatchScore.from_paragraphs_features(main_paragraph, paragraph)
            if alignment_score.overall_score < THRESHOLD[-1] - 0.3:
                continue

            self._alignment_scores[main_paragraph] = AlignmentScore(
                main_paragraph=main_paragraph, other_paragraph=paragraph, score=alignment_score.overall_score
            )

            aligned_paragraphs[idx] = paragraph

    def assign_unmatch_paragraphs(self):
        inverse_alignment_scores = {score.other_paragraph: score for score in self._alignment_scores.values()}
        for idx, paragraph_to_be_merged in enumerate(self.paragraphs):
            if paragraph_to_be_merged in inverse_alignment_scores:
                continue

            previous_paragraph = self.paragraphs[idx - 1] if idx > 0 else None
            if previous_paragraph in inverse_alignment_scores:
                main_to_receive = inverse_alignment_scores[previous_paragraph].main_paragraph
                merged_paragraph = previous_paragraph.model_copy().merge(paragraph_to_be_merged)
                if self.should_merge_paragraphs(main_to_receive, merged_paragraph):
                    previous_paragraph.merge(paragraph_to_be_merged)
                    continue

            next_paragraph = self.paragraphs[idx + 1] if idx + 1 < len(self.paragraphs) else None
            if next_paragraph not in inverse_alignment_scores:
                continue

            main_to_receive = inverse_alignment_scores[next_paragraph].main_paragraph
            merged_paragraph = paragraph_to_be_merged.model_copy().merge(next_paragraph)
            if self.should_merge_paragraphs(main_to_receive, merged_paragraph):
                paragraph_to_be_merged.merge(next_paragraph)

    def should_merge_paragraphs(self, main_paragraph: ParagraphFeatures, merged_paragraph: ParagraphFeatures) -> bool:
        if main_paragraph not in self._alignment_scores:
            return False

        previous_score = self._alignment_scores[main_paragraph].score
        match_score = ParagraphMatchScore.from_paragraphs_features(main_paragraph, merged_paragraph)
        return previous_score <= match_score.overall_score
