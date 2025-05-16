import re
from math import ceil

from pdf_token_type_labels.TokenType import TokenType
from pydantic import BaseModel
from rapidfuzz import fuzz

from multilingual_paragraph_extractor.domain.AlignmentScore import AlignmentScore
from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures
from multilingual_paragraph_extractor.domain.ParagraphMatchScore import ParagraphMatchScore

BLOCK_SIZE = 50
THRESHOLD = [0.9, 0.86, 0.82, 0.78, 0.74, 0.7]
HEADER_SIMILARITY_THRESHOLD = 90
TOP_OF_PAGE_THRESHOLD = 0.2
REPEATED_HEADER_THRESHOLD = 0.2


class ParagraphsFromLanguage(BaseModel):
    language: str
    paragraphs: list[ParagraphFeatures]
    is_main_language: bool
    _aligned_paragraphs: list[ParagraphFeatures] = list()
    _alignment_scores: dict[ParagraphFeatures, AlignmentScore] = dict()
    _main_language_paragraphs: list[ParagraphFeatures] = list()

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
        segmentation_changed = self.fix_other_language_segmentation()
        segmentation_changed = self.fix_main_language_when_other_language_not_assigned() or segmentation_changed
        segmentation_changed = self.fix_main_language_when_main_language_not_assigned() or segmentation_changed
        return segmentation_changed

    def remove_no_text_types(self):
        text_content_types = [
            TokenType.LIST_ITEM,
            TokenType.TEXT,
        ]
        self.paragraphs = [x for x in self.paragraphs if x.paragraph_type in text_content_types]

    def merge_colliding_segments(self):
        fixed_paragraphs = []
        index = 0

        while index < len(self.paragraphs):
            paragraph = self.paragraphs[index]

            if index + 1 >= len(self.paragraphs):
                fixed_paragraphs.append(paragraph)
                index += 1
                continue

            if paragraph.collide(self.paragraphs[index + 1]):
                merged_segment = paragraph.merge(self.paragraphs[index + 1])
                fixed_paragraphs.append(merged_segment)
                index += 2
                continue

            fixed_paragraphs.append(paragraph)
            index += 1

        self.paragraphs = fixed_paragraphs

    def remove_no_text_paragraphs(self):
        cleaned_paragraphs = list()
        for paragraph in self.paragraphs:
            if not paragraph.text_cleaned:
                continue

            if not any(char.isalnum() for char in paragraph.text_cleaned):
                continue

            regular_characters_regex = (
                r"[^a-zA-Z0-9\sа-яА-Яά-ωΑ-Ω\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
            )
            regular_characters = re.sub(regular_characters_regex, "", paragraph.text_cleaned)

            if len(regular_characters) <= 1:
                continue

            cleaned_paragraphs.append(paragraph)

        return cleaned_paragraphs

    def remove_duplicated_text(self):
        cleaned_paragraphs = list()
        for paragraph, next_paragraph in zip(self.paragraphs, self.paragraphs[1:]):
            if paragraph.text_cleaned == next_paragraph.text_cleaned:
                continue

            cleaned_paragraphs.append(paragraph)

        if self.paragraphs:
            cleaned_paragraphs.append(self.paragraphs[-1])

        return cleaned_paragraphs

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
        pages_number = max([x.page_number for x in self.paragraphs]) if self.paragraphs else 1
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
        text = text.strip()

        # General pattern for common list markers
        patterns = [
            # Numbers with different decorators: 1. 1) (1) 1- etc.
            r"^\d+[\.\)\-]?\d*$",
            r"^\(\d+\)$",
            # Letters with different decorators: a. a) (a) A. A) (A) etc.
            r"^[a-zA-Z][\.\)\-]?$",
            r"^\([a-zA-Z]\)$",
            # Roman numerals (both cases) with different decorators: i. I. (i) (I) etc.
            r"^(?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv)[\.\)\-]?$",
            r"^(?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV)[\.\)\-]?$",
            r"^\((?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv)\)$",
            r"^\((?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV)\)$",
            # Common bullet points and decorative markers
            r"^[-–—•∙◦○●\*\+]$",
            # Square brackets: [1] [a] etc.
            r"^\[\d+\]$",
            r"^\[[a-zA-Z]\]$",
            # Additional common separators
            r"^§\s*\d+$",  # Section symbol with number
            r"^¶\s*\d+$",  # Paragraph symbol with number
        ]

        return any(re.match(pattern, text, re.IGNORECASE) for pattern in patterns)

    def fix_other_language_segmentation(self):
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

        # Stores final idx1 -> idx2 for all successful matches
        indexes_matching: dict[int, int] = dict()
        # Cache for (paragraph_feature1, paragraph_feature2) -> score
        scores_cache: dict[tuple[ParagraphFeatures, ParagraphFeatures], float] = dict()

        # Ensure thresholds are processed from high to low
        # The original THRESHOLD constant is assumed to be iterable.
        sorted_thresholds = sorted(THRESHOLD, reverse=True)

        for threshold_index, threshold in enumerate(sorted_thresholds):
            # This variable tracks the 'center' for the sliding window search (first pass)
            # or is updated by confirmed matches. It's reset for each threshold pass.
            current_pass_search_pivot_idx2 = 0

            # Iterate over all main language paragraph indices to correctly handle
            # current_pass_search_pivot_idx2 and process paragraphs in their natural order.
            for idx1 in range(len(self._main_language_paragraphs)):
                if idx1 not in unmatched_1: # If already matched (e.g., by a higher threshold)
                    # If this paragraph (idx1) IS matched, its corresponding idx2 can serve as
                    # the pivot for the *next* paragraph (idx1+1) in the first threshold pass.
                    if idx1 in indexes_matching: # Check if current idx1 *is* a committed match
                         current_pass_search_pivot_idx2 = indexes_matching[idx1]
                    continue # Move to the next idx1

                # At this point, idx1 is in unmatched_1 and needs processing for the current threshold.

                # Determine the search pivot for the current idx1's window (if first pass).
                # This is either inherited from the previous idx1's processing or set by idx1-1's match.
                pivot_for_idx1_search = current_pass_search_pivot_idx2
                if idx1 > 0 and (idx1 - 1) in indexes_matching:
                    pivot_for_idx1_search = indexes_matching[idx1 - 1]

                potential_matches_idx2_list: list[int] = []

                if threshold_index == 0: # Highest threshold: use original windowing logic
                    start_j_window = max(0, pivot_for_idx1_search - BLOCK_SIZE)
                    end_j_window = min(len(self.paragraphs), pivot_for_idx1_search + BLOCK_SIZE)


                    window_candidates_idx2 = list(unmatched_2 & set(range(start_j_window, end_j_window)))

                    after_pivot = sorted([p_idx for p_idx in window_candidates_idx2 if p_idx > pivot_for_idx1_search])
                    before_pivot = sorted([p_idx for p_idx in window_candidates_idx2 if p_idx <= pivot_for_idx1_search], reverse=True)
                    potential_matches_idx2_list = after_pivot + before_pivot

                else: # Lower thresholds: search only in gaps defined by `indexes_matching`
                    sorted_matched_main_indices = sorted(indexes_matching.keys())

                    prev_anchor_main_idx = -1
                    for k in sorted_matched_main_indices:
                        if k < idx1:
                            prev_anchor_main_idx = k
                        else:
                            break

                    next_anchor_main_idx = -1
                    for k in sorted_matched_main_indices:
                        if k > idx1:
                            next_anchor_main_idx = k
                            break

                    gap_search_start_idx2 = 0
                    if prev_anchor_main_idx != -1:
                        gap_search_start_idx2 = indexes_matching[prev_anchor_main_idx] + 1

                    gap_search_end_idx2 = len(self.paragraphs)
                    if next_anchor_main_idx != -1:
                        gap_search_end_idx2 = indexes_matching[next_anchor_main_idx]

                    if gap_search_start_idx2 < gap_search_end_idx2:
                        gap_candidates_idx2 = list(unmatched_2 & set(range(gap_search_start_idx2, gap_search_end_idx2)))
                        potential_matches_idx2_list = sorted(gap_candidates_idx2)
                    else:
                        potential_matches_idx2_list = []

                best_match_this_idx1_idx2 = None
                current_best_score_for_idx1 = threshold

                for idx2_candidate in potential_matches_idx2_list:
                    para1_features = self._main_language_paragraphs[idx1]
                    para2_features = self.paragraphs[idx2_candidate]

                    if (para1_features, para2_features) in scores_cache:
                        score = scores_cache[(para1_features, para2_features)]
                    else:
                        match_obj = ParagraphMatchScore.from_paragraphs_features(
                            para1_features, para2_features
                        )
                        score = match_obj.overall_score
                        scores_cache[(para1_features, para2_features)] = score

                    if score > current_best_score_for_idx1:
                        current_best_score_for_idx1 = score
                        best_match_this_idx1_idx2 = idx2_candidate
                        if score > 0.95: # High-confidence optimization
                            break

                if best_match_this_idx1_idx2 is not None:
                    indexes_matching[idx1] = best_match_this_idx1_idx2

                    alignment_score_obj = AlignmentScore(
                        main_paragraph=self._main_language_paragraphs[idx1],
                        other_paragraph=self.paragraphs[best_match_this_idx1_idx2],
                        score=current_best_score_for_idx1,
                    )
                    self._alignment_scores[self._main_language_paragraphs[idx1]] = alignment_score_obj

                    unmatched_1.remove(idx1)
                    if best_match_this_idx1_idx2 in unmatched_2: # Should be true
                        unmatched_2.remove(best_match_this_idx1_idx2)

                    current_pass_search_pivot_idx2 = best_match_this_idx1_idx2
                else: # No match found for idx1 at this threshold
                    if threshold_index == 0: # Only adjust pivot this way in the first (windowed) pass
                        current_pass_search_pivot_idx2 = pivot_for_idx1_search + 1
                        if len(self.paragraphs) > 0:
                            current_pass_search_pivot_idx2 = min(current_pass_search_pivot_idx2, len(self.paragraphs) - 1)
                        else:
                            current_pass_search_pivot_idx2 = 0
                        current_pass_search_pivot_idx2 = max(0, current_pass_search_pivot_idx2)

        # self._alignment_scores is now populated
        # No explicit return needed if it's modifying a class member

    def is_same_pdf(self):
        paragraph_count = len(self._main_language_paragraphs)
        if not paragraph_count:
            return True
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

    def fix_main_language_when_other_language_not_assigned(self):
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

    def fix_main_language_when_main_language_not_assigned(self):
        main_paragraphs_count = len(self._main_language_paragraphs)
        paragraphs_to_be_remove = list()
        for paragraph_to_be_merged in reversed(self._main_language_paragraphs):
            if paragraph_to_be_merged in self._alignment_scores:
                continue

            idx = self._main_language_paragraphs.index(paragraph_to_be_merged)
            previous_paragraph = self._main_language_paragraphs[idx - 1] if idx > 0 else None
            if previous_paragraph in self._alignment_scores:
                score = self._alignment_scores[previous_paragraph].score
                other_to_compare = self._alignment_scores[previous_paragraph].other_paragraph
                if self.should_merge_paragraphs(other_to_compare, score, previous_paragraph, paragraph_to_be_merged):
                    merged = previous_paragraph.merge(paragraph_to_be_merged)
                    self._main_language_paragraphs[idx - 1] = merged
                    self._alignment_scores[merged] = AlignmentScore(
                        main_paragraph=merged,
                        other_paragraph=other_to_compare,
                        score=score,
                    )
                    paragraphs_to_be_remove.append(paragraph_to_be_merged)
                    continue

            next_paragraph = (
                self._main_language_paragraphs[idx + 1] if idx + 1 < len(self._main_language_paragraphs) else None
            )
            if next_paragraph not in self._alignment_scores:
                continue
            other_to_compare = self._alignment_scores[next_paragraph].other_paragraph
            score = self._alignment_scores[next_paragraph].score
            if self.should_merge_paragraphs(other_to_compare, score, paragraph_to_be_merged, next_paragraph):
                self._main_language_paragraphs[idx + 1] = paragraph_to_be_merged.merge(next_paragraph)
                paragraphs_to_be_remove.append(next_paragraph)

        for paragraph in paragraphs_to_be_remove:
            if paragraph in self._main_language_paragraphs:
                self._main_language_paragraphs.remove(paragraph)

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

    def remove_big_no_text_paragraphs(self):
        threshold_area = 0.2 * self.paragraphs[0].page_width * self.paragraphs[0].page_height if self.paragraphs else 0
        fixed_paragraphs = list()

        for paragraph in self.paragraphs:
            if not len(paragraph.original_text):
                continue

            if paragraph.bounding_box.area() < threshold_area:
                fixed_paragraphs.append(paragraph)
                continue
            if paragraph.font.font_size > 10:
                font_size_corrector = 1 + abs(paragraph.font.font_size - 10) / 10
            else:
                font_size_corrector = 1 - abs(paragraph.font.font_size - 10) / 10

            if paragraph.bounding_box.area() / (len(paragraph.original_text) * font_size_corrector) > 100:
                continue

            fixed_paragraphs.append(paragraph)

        self.paragraphs = fixed_paragraphs

    def to_db(self):
        return ParagraphsFromLanguage(
            language=self.language,
            paragraphs=[x.to_db() for x in self.paragraphs],
            is_main_language=self.is_main_language,
        )
