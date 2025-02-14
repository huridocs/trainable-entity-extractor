import re
from math import ceil

from pdf_token_type_labels.TokenType import TokenType
from pydantic import BaseModel
from rapidfuzz import fuzz

from multilingual_paragraph_extractor.domain.AlignmentScore import AlignmentScore
from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures
from multilingual_paragraph_extractor.domain.ParagraphMatchScore import ParagraphMatchScore

BLOCK_SIZE = 10
THRESHOLD = [0.9, 0.86, 0.82, 0.78]
HEADER_SIMILARITY_THRESHOLD = 90
TOP_OF_PAGE_THRESHOLD = 0.1
REPEATED_HEADER_THRESHOLD = 0.2


class ParagraphsFromLanguage(BaseModel):
    language: str
    paragraphs: list[ParagraphFeatures]
    is_main_language: bool
    _aligned_paragraphs: list[ParagraphFeatures] = list()
    _alignment_scores: dict[ParagraphFeatures, AlignmentScore] = dict()
    _main_language_paragraphs = list[ParagraphFeatures]

    class Config:
        arbitrary_types_allowed = True

    def replace_paragraphs_to_aligned(self):
        self.paragraphs = self._aligned_paragraphs

    def set_as_main_language(self):
        self.is_main_language = True
        self._aligned_paragraphs = self.paragraphs

    def align(self, main_language: "ParagraphsFromLanguage"):
        self._main_language_paragraphs = main_language.paragraphs
        self.set_alignment_scores()
        self.get_aligned_paragraphs_from_scores()

        if not self.is_same_pdf():
            self._aligned_paragraphs = [ParagraphFeatures.get_empty() for _ in range(len(self._main_language_paragraphs))]
            return

        self.assign_missing_in_both_languages()

    def fix_segments(self, main_language: "ParagraphsFromLanguage") -> bool:
        self.align(main_language)
        segmentation_fixed = self.fix_segmentation()
        if segmentation_fixed:
            self.align(main_language)
        main_segmentation_fixed = self.assign_unmatch_paragraphs_merging_it_with_other()
        if segmentation_fixed or main_segmentation_fixed:
            return True
        return False

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
        header_paragraphs = self.find_headers_with_similarities()
        self.paragraphs = [x for x in self.paragraphs if x.paragraph_type not in types]
        self.paragraphs = [p for p in self.paragraphs if p not in header_paragraphs]

    @staticmethod
    def is_top_or_bottom_of_page(paragraph: ParagraphFeatures, page_height: int):
        on_top = paragraph.bounding_box.top < page_height * TOP_OF_PAGE_THRESHOLD
        on_bottom = paragraph.bounding_box.bottom > page_height * (1 - TOP_OF_PAGE_THRESHOLD)
        return on_top or on_bottom

    def find_headers_with_similarities(self):
        paragraphs_on_top = [x for x in self.paragraphs if self.is_top_or_bottom_of_page(x, self.paragraphs[0].page_height)]
        pages_number = max([x.page_number for x in self.paragraphs])
        headers = {}
        for paragraph in paragraphs_on_top:
            found_match = False
            for header_text in headers:
                if fuzz.ratio(paragraph.text_cleaned, header_text) > HEADER_SIMILARITY_THRESHOLD:
                    headers[header_text].append(paragraph)
                    found_match = True
                    break
            if not found_match:
                headers[paragraph.text_cleaned] = [paragraph]

        min_pages = max(ceil(pages_number * REPEATED_HEADER_THRESHOLD), 3)
        repeated_headers = {k: v for k, v in headers.items() if len(v) >= min_pages}
        header_paragraphs = [p for header_list in repeated_headers.values() for p in header_list]
        return header_paragraphs

    @staticmethod
    def is_paragraph_separators(text: str) -> bool:
        patterns = [
            r"(?:^|\s+)(\d+)\.",  # Matches numbers followed by a dot
            r"(?:^|\s+)(\d+)",  # Matches numbers
            r"(?:^|\s+)\((\d+)\)",  # Matches numbers inside parentheses
            r"(?:^|\s+)([a-z])\)",  # Matches lowercase letters followed by a closing parenthesis
            r"(?:^|\s+)([A-Z])\)",  # Matches uppercase letters followed by a closing parenthesis
            r"^[a-fA-F]$",  # Matches single characters a-f or A-F
            r"(?:^|\s+)([ivx]+)",  # Matches Roman numerals in lowercase
            r"(?:^|\s+)([IVXLC]+)",  # Matches Roman numerals in uppercase
            r"(?:^|\s+)([•*•-])",  # Matches bullet points
        ]

        for pattern in patterns:
            for _ in re.finditer(pattern, text):
                return True

        return False

    def fix_segmentation(self):
        previous_paragraph_count = len(self.paragraphs)
        for main_unassigned_paragraph in reversed(self._main_language_paragraphs):
            if main_unassigned_paragraph in self._alignment_scores:
                continue
            idx = self._main_language_paragraphs.index(main_unassigned_paragraph)
            previous_main_paragraph = self._main_language_paragraphs[idx - 1] if idx > 0 else None
            if previous_main_paragraph in self._alignment_scores:
                to_receive = self._alignment_scores[previous_main_paragraph].other_paragraph
                score = self._alignment_scores[previous_main_paragraph].score
                if self.should_merge_paragraphs(to_receive, score, previous_main_paragraph, main_unassigned_paragraph):
                    self.split_paragraph(self.paragraphs, main_unassigned_paragraph, to_receive)
                    continue

            next_paragraph = (
                self._main_language_paragraphs[idx + 1] if idx + 1 < len(self._main_language_paragraphs) else None
            )
            if next_paragraph not in self._alignment_scores:
                continue

            to_receive = self._alignment_scores[next_paragraph].main_paragraph
            score = self._alignment_scores[next_paragraph].score
            if self.should_merge_paragraphs(to_receive, score, main_unassigned_paragraph, next_paragraph):
                self.split_paragraph(self.paragraphs, next_paragraph, to_receive)

        return len(self.paragraphs) != previous_paragraph_count

    @staticmethod
    def split_paragraph(paragraph_list: list[ParagraphFeatures], next_main: ParagraphFeatures, to_fix: ParagraphFeatures):
        splitter_word = next_main.first_word

        if not ParagraphsFromLanguage.is_paragraph_separators(splitter_word):
            return False

        if splitter_word not in to_fix.original_text:
            return False

        if to_fix.original_text.count(splitter_word) > 1:
            return False

        if to_fix.original_text.strip().startswith(splitter_word):
            return False

        if to_fix.original_text.strip().endswith(splitter_word):
            return False

        try:
            index = paragraph_list.index(to_fix)
        except ValueError:
            return False

        paragraph_1, paragraph_2 = to_fix.split_paragraph(splitter_word)
        paragraph_list[index] = paragraph_1
        paragraph_list.insert(index + 1, paragraph_2)
        return True

    def merge_paragraphs_spanning_two_pages(self):
        fixed_paragraphs = []
        index = 0

        while index < len(self.paragraphs):
            paragraph = self.paragraphs[index]

            if index + 1 >= len(self.paragraphs):
                fixed_paragraphs.append(paragraph)
                index += 1
                continue

            if paragraph.is_part_of_same_segment(self.paragraphs[index + 1]):
                merged_segment = paragraph.merge(self.paragraphs[index + 1])
                fixed_paragraphs.append(merged_segment)
                index += 2
                continue

            fixed_paragraphs.append(paragraph)
            index += 1

        self.paragraphs = fixed_paragraphs

    def set_alignment_scores(self):
        self._alignment_scores = dict()
        unmatched_1 = set(range(len(self._main_language_paragraphs)))
        unmatched_2 = set(range(len(self.paragraphs)))

        indexes_matching: dict[int, int] = dict()
        scores: dict[(ParagraphFeatures, ParagraphFeatures), float] = dict()

        for threshold in THRESHOLD:
            last_idx2_inserted = 0
            threshold_block_size = BLOCK_SIZE

            for idx1 in list(unmatched_1):
                if idx1 - 1 in indexes_matching:
                    last_idx2_inserted = indexes_matching[idx1 - 1]
                start_j = max(0, last_idx2_inserted - threshold_block_size)
                end_j = min(len(self.paragraphs), last_idx2_inserted + threshold_block_size)
                current_block2 = list(unmatched_2 & set(range(start_j, end_j)))

                after_indexes = sorted([x for x in current_block2 if x > last_idx2_inserted])
                before_indexes = sorted([x for x in current_block2 if x not in after_indexes], reverse=True)
                current_block2 = after_indexes + before_indexes

                best_match = None
                best_score = threshold

                for idx2 in current_block2:
                    if (self._main_language_paragraphs[idx1], self.paragraphs[idx2]) in scores:
                        score = scores[(self._main_language_paragraphs[idx1], self.paragraphs[idx2])]
                    else:
                        match = ParagraphMatchScore.from_paragraphs_features(
                            self._main_language_paragraphs[idx1], self.paragraphs[idx2]
                        )
                        score = match.overall_score
                        scores[(self._main_language_paragraphs[idx1], self.paragraphs[idx2])] = score

                    main_first_word = self._main_language_paragraphs[idx1].first_word
                    other_first_word = self.paragraphs[idx2].first_word
                    if score > best_score:
                        best_score = score
                        best_match = idx2
                        if score > 0.95:
                            break

                if best_match is not None:
                    last_idx2_inserted = best_match
                    indexes_matching[idx1] = best_match
                    alignment_score = AlignmentScore(
                        main_paragraph=self._main_language_paragraphs[idx1],
                        other_paragraph=self.paragraphs[best_match],
                        score=best_score,
                    )
                    self._alignment_scores[self._main_language_paragraphs[idx1]] = alignment_score
                    unmatched_2.remove(best_match)
                    unmatched_1.remove(idx1)
                else:
                    last_idx2_inserted += 1

    def is_same_pdf(self):
        paragraph_count = len(self._main_language_paragraphs)
        unmatched_paragraphs = [x for x in self._main_language_paragraphs if x not in self._alignment_scores]
        match_percentage = 100 * (paragraph_count - len(unmatched_paragraphs)) / paragraph_count
        return 50 < match_percentage

    def get_aligned_paragraphs_from_scores(self):
        self._aligned_paragraphs: list[ParagraphFeatures] = []
        for paragraph in self._main_language_paragraphs:
            if paragraph in self._alignment_scores:
                paragraph_to_add = self._alignment_scores[paragraph].other_paragraph
            else:
                paragraph_to_add = ParagraphFeatures.get_empty()

            self._aligned_paragraphs.append(paragraph_to_add)

    def assign_missing_in_both_languages(self):
        for idx, main_paragraph in enumerate(self._main_language_paragraphs):
            if main_paragraph in self._alignment_scores:
                continue

            main_previous = self._main_language_paragraphs[idx - 1] if idx > 0 else None
            if main_previous not in self._alignment_scores:
                continue

            previous_index = self.paragraphs.index(self._alignment_scores[main_previous].other_paragraph)
            paragraph = self.paragraphs[previous_index + 1] if previous_index + 1 < len(self.paragraphs) else None
            if not paragraph or paragraph in self._aligned_paragraphs:
                continue

            alignment_score = ParagraphMatchScore.from_paragraphs_features(main_paragraph, paragraph)
            if alignment_score.overall_score < THRESHOLD[-1] - 0.3:
                continue

            self._alignment_scores[main_paragraph] = AlignmentScore(
                main_paragraph=main_paragraph, other_paragraph=paragraph, score=alignment_score.overall_score
            )

            self._aligned_paragraphs[idx] = paragraph

    def assign_unmatch_paragraphs_merging_it_with_other(self):
        inverse_alignment_scores = {score.other_paragraph: score for score in self._alignment_scores.values()}
        main_paragraphs_count = len(self._main_language_paragraphs)
        paragraphs_to_be_remove = list()
        for paragraph_to_be_merged in reversed(self.paragraphs):
            if paragraph_to_be_merged in inverse_alignment_scores:
                continue

            idx = self.paragraphs.index(paragraph_to_be_merged)
            previous_paragraph = self.paragraphs[idx - 1] if idx > 0 else None
            if previous_paragraph in inverse_alignment_scores:
                main_to_receive = inverse_alignment_scores[previous_paragraph].main_paragraph
                score = inverse_alignment_scores[previous_paragraph].score
                if self.should_merge_paragraphs(main_to_receive, score, previous_paragraph, paragraph_to_be_merged):
                    to_remove = self.split_main_or_merge_other(main_to_receive, previous_paragraph, paragraph_to_be_merged)
                    paragraphs_to_be_remove.extend(to_remove)
                    continue

            next_paragraph = self.paragraphs[idx + 1] if idx + 1 < len(self.paragraphs) else None
            if next_paragraph not in inverse_alignment_scores:
                continue
            main_to_receive = inverse_alignment_scores[next_paragraph].main_paragraph
            score = inverse_alignment_scores[next_paragraph].score
            if self.should_merge_paragraphs(main_to_receive, score, paragraph_to_be_merged, next_paragraph):
                to_remove = self.split_main_or_merge_other(main_to_receive, paragraph_to_be_merged, next_paragraph)
                paragraphs_to_be_remove.extend(to_remove)

        for paragraph in paragraphs_to_be_remove:
            if paragraph in self.paragraphs:
                self.paragraphs.remove(paragraph)

        return len(paragraphs_to_be_remove) != 0 or main_paragraphs_count != len(self._main_language_paragraphs)

    def split_main_or_merge_other(
        self, main_paragraph: ParagraphFeatures, previous_paragraph: ParagraphFeatures, next_paragraph: ParagraphFeatures
    ):
        if self.split_paragraph(self._main_language_paragraphs, next_paragraph, main_paragraph):
            return []

        if previous_paragraph.get_distance(next_paragraph) > 0.02:
            return []

        if previous_paragraph in self._aligned_paragraphs:
            previous_paragraph.merge(next_paragraph)
            return [next_paragraph]

        if next_paragraph in self._aligned_paragraphs:
            previous_paragraph.merge(next_paragraph)
            index = self.paragraphs.index(next_paragraph)
            self.paragraphs[index] = previous_paragraph
            return [next_paragraph]

        return []

    @staticmethod
    def should_merge_paragraphs(
        paragraph: ParagraphFeatures,
        previous_score: float,
        previous_paragraph_to_merge: ParagraphFeatures,
        next_paragraph_to_merge: ParagraphFeatures,
    ) -> bool:
        merged_paragraph = previous_paragraph_to_merge.model_copy(deep=True).merge(next_paragraph_to_merge)
        match_score = ParagraphMatchScore.from_paragraphs_features(paragraph, merged_paragraph)
        return previous_score <= match_score.overall_score

    def is_aligned(self, main_language: "ParagraphsFromLanguage") -> bool:
        if not self._aligned_paragraphs:
            return False
        return len(self._aligned_paragraphs) == len(main_language.paragraphs)
