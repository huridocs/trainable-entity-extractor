import re

from pdf_token_type_labels.TokenType import TokenType
from pydantic import BaseModel
from rapidfuzz import fuzz

from multilingual_paragraph_extractor.domain.AlignmentScore import AlignmentScore
from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures
from multilingual_paragraph_extractor.domain.ParagraphMatchScore import ParagraphMatchScore

BLOCK_SIZE = 10
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
        header_paragraphs = self.find_headers_with_similarities()
        self.paragraphs = [x for x in self.paragraphs if x.paragraph_type not in types]
        self.paragraphs = [p for p in self.paragraphs if p not in header_paragraphs]

    @staticmethod
    def is_top_of_page(paragraph: ParagraphFeatures, page_height: int, threshold: int = 0.1):
        return paragraph.bounding_box.top < page_height * threshold

    def find_headers_with_similarities(self, similarity_threshold: int = 90):
        paragraphs_on_top = [x for x in self.paragraphs if self.is_top_of_page(x, self.paragraphs[0].page_height)]

        headers = {}
        for paragraph in paragraphs_on_top:
            found_match = False
            for header_text in headers:
                if fuzz.ratio(paragraph.text_cleaned, header_text) > similarity_threshold:
                    headers[header_text].append(paragraph)
                    found_match = True
                    break
            if not found_match:
                headers[paragraph.text_cleaned] = [paragraph]

        repeated_headers = {k: v for k, v in headers.items() if len(v) > 1}
        header_paragraphs = [p for header_list in repeated_headers.values() for p in header_list]
        return header_paragraphs

    @staticmethod
    def find_paragraph_separators(text: str) -> list[tuple[int, str]]:
        patterns = [
            r"(?:^|\s+)(\d+)\.\s",
            r"(?:^|\s+)\((\d+)\)\s",
            r"(?:^|\s+)([a-z])\)\s",
            r"(?:^|\s+)([ivx]+)\)\s",
            r"(?:^|\s+)([IVXLC]+)\.\s",
        ]

        separation_points = []

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                separation_points.append((match.start(), match.group(0).strip()))
        return sorted(separation_points)

    @staticmethod
    def split_merged_paragraph(
        paragraph: ParagraphFeatures, split_paragraphs_start_indexes: list[int]
    ) -> list[ParagraphFeatures]:
        split_paragraphs = []
        for i in range(len(split_paragraphs_start_indexes)):
            start_index = split_paragraphs_start_indexes[i]
            if i == len(split_paragraphs_start_indexes) - 1:
                end_index = len(paragraph.original_text)
            else:
                end_index = split_paragraphs_start_indexes[i + 1]
            original_text = paragraph.original_text[start_index:end_index].strip()
            new_paragraph = ParagraphFeatures.get_split_paragraph(paragraph, original_text)
            split_paragraphs.append(new_paragraph)
        return split_paragraphs

    def split_main_paragraphs(self, main_paragraphs, aligned_paragraphs):
        paragraphs_to_remove = []
        for i, main_paragraph in enumerate(main_paragraphs[:-1]):
            if main_paragraph not in self._alignment_scores:
                continue
            aligned_paragraph = self._alignment_scores[main_paragraph].other_paragraph
            next_paragraph_index = self.paragraphs.index(aligned_paragraph) + 1

            if next_paragraph_index >= len(self.paragraphs):
                continue

            if self.paragraphs[next_paragraph_index] in aligned_paragraphs:
                continue

            next_paragraph_indexes = [next_paragraph_index - 1, next_paragraph_index]
            for index in range(next_paragraph_index + 1, len(self.paragraphs)):
                if self.paragraphs[index] in aligned_paragraphs:
                    break
                next_paragraph_indexes.append(index)

            main_paragraph_separators = self.find_paragraph_separators(main_paragraph.original_text)
            other_paragraph_separators = self.find_paragraph_separators(aligned_paragraph.original_text)

            if [s[1] for s in main_paragraph_separators] == [s[1] for s in other_paragraph_separators]:
                continue

            alignment_score = self._alignment_scores[main_paragraph].score

            split_paragraphs_start_indexes = [
                s[0]
                for index, s in enumerate(main_paragraph_separators)
                if s[1] in self.paragraphs[next_paragraph_indexes[index]].first_word
            ]
            split_paragraphs = self.split_merged_paragraph(main_paragraph, split_paragraphs_start_indexes)

            if len(split_paragraphs) != len(next_paragraph_indexes):
                continue

            main_paragraph_insert_index = main_paragraphs.index(main_paragraph)
            aligned_paragraph_insert_index = aligned_paragraphs.index(aligned_paragraph) + 1

            for index in range(len(split_paragraphs)):
                alignment = AlignmentScore(
                    main_paragraph=split_paragraphs[index],
                    other_paragraph=self.paragraphs[next_paragraph_indexes[index]],
                    score=alignment_score,
                )
                self._alignment_scores[split_paragraphs[index]] = alignment
                main_paragraphs.insert(main_paragraph_insert_index, split_paragraphs[index])
                aligned_paragraphs.insert(aligned_paragraph_insert_index, self.paragraphs[next_paragraph_indexes[index]])
                main_paragraph_insert_index += 1
                aligned_paragraph_insert_index += 1

            paragraphs_to_remove.append(main_paragraph)

        for paragraph in paragraphs_to_remove:
            del self._alignment_scores[paragraph]
            main_paragraphs.remove(paragraph)

    def split_other_paragraphs(self, main_paragraphs, aligned_paragraphs):
        paragraphs_to_remove = []
        for i, main_paragraph in enumerate(main_paragraphs[:-1]):
            if main_paragraph in self._alignment_scores:
                continue

            previous_paragraph_index = i - 1
            if previous_paragraph_index < 0:
                continue

            previous_paragraph = main_paragraphs[previous_paragraph_index]
            if previous_paragraph not in self._alignment_scores:
                continue

            main_paragraph_text = previous_paragraph.original_text
            unmatched_main_paragraphs = [previous_paragraph]

            for i in range(main_paragraphs.index(main_paragraph), len(main_paragraphs)):
                if main_paragraphs[i] in self._alignment_scores:
                    break
                unmatched_main_paragraphs.append(main_paragraphs[i])
                main_paragraph_text += " " + main_paragraphs[i].original_text

            other_paragraph = self._alignment_scores[previous_paragraph].other_paragraph
            main_paragraph_separators = self.find_paragraph_separators(main_paragraph_text)
            other_paragraph_separators = self.find_paragraph_separators(other_paragraph.original_text)

            if not main_paragraph_separators:
                continue

            if [s[1] for s in main_paragraph_separators] != [s[1] for s in other_paragraph_separators]:
                continue

            alignment_score = self._alignment_scores[previous_paragraph].score
            del self._alignment_scores[previous_paragraph]

            split_paragraphs_start_indexes = [
                s[0]
                for index, s in enumerate(other_paragraph_separators)
                if s[1] in unmatched_main_paragraphs[index].first_word
            ]
            split_paragraphs = self.split_merged_paragraph(other_paragraph, split_paragraphs_start_indexes)
            other_paragraph_insert_index = self.paragraphs.index(other_paragraph)
            other_paragraph_alignment_index = aligned_paragraphs.index(other_paragraph)
            paragraphs_to_remove.append(other_paragraph)

            for index, split_paragraph in enumerate(split_paragraphs):
                self.paragraphs.insert(other_paragraph_insert_index, split_paragraph)
                aligned_paragraphs.insert(other_paragraph_alignment_index, split_paragraph)

                alignment = AlignmentScore(
                    main_paragraph=unmatched_main_paragraphs[index], other_paragraph=split_paragraph, score=alignment_score
                )
                self._alignment_scores[unmatched_main_paragraphs[index]] = alignment

                other_paragraph_insert_index += 1
                other_paragraph_alignment_index += 1

        for paragraph in paragraphs_to_remove:
            self.paragraphs.remove(paragraph)
            aligned_paragraphs.remove(paragraph)

    def split_merged_paragraphs(self, main_paragraphs: list[ParagraphFeatures], aligned_paragraphs):
        self.split_main_paragraphs(main_paragraphs, aligned_paragraphs)
        self.split_other_paragraphs(main_paragraphs, aligned_paragraphs)

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
                    if (main_paragraphs[idx1], self.paragraphs[idx2]) in scores:
                        score = scores[(main_paragraphs[idx1], self.paragraphs[idx2])]
                    else:
                        match = ParagraphMatchScore.from_paragraphs_features(main_paragraphs[idx1], self.paragraphs[idx2])
                        score = match.overall_score
                        scores[(main_paragraphs[idx1], self.paragraphs[idx2])] = score

                    main_first_word = main_paragraphs[idx1].first_word
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
                        main_paragraph=main_paragraphs[idx1], other_paragraph=self.paragraphs[best_match], score=best_score
                    )
                    self._alignment_scores[main_paragraphs[idx1]] = alignment_score
                    unmatched_2.remove(best_match)
                    unmatched_1.remove(idx1)
                else:
                    last_idx2_inserted += 1

    def align(self, main_language: "ParagraphsFromLanguage"):
        self.set_alignment_scores(main_language.paragraphs)
        aligned_paragraphs = self.get_aligned_paragraphs_from_scores(main_language)
        self.assign_missing_in_both_languages(main_language.paragraphs, aligned_paragraphs)
        self.assign_unmatch_paragraphs()
        self.split_merged_paragraphs(main_language.paragraphs, aligned_paragraphs)
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
                if self.should_merge_paragraphs(main_to_receive, previous_paragraph, paragraph_to_be_merged):
                    previous_paragraph.merge(paragraph_to_be_merged)
                    continue

            next_paragraph = self.paragraphs[idx + 1] if idx + 1 < len(self.paragraphs) else None
            if next_paragraph not in inverse_alignment_scores:
                continue
            main_to_receive = inverse_alignment_scores[next_paragraph].main_paragraph
            if self.should_merge_paragraphs(main_to_receive, paragraph_to_be_merged, next_paragraph):
                paragraph_to_be_merged.merge(next_paragraph)

    def should_merge_paragraphs(
        self, main_paragraph: ParagraphFeatures, previous_paragraph: ParagraphFeatures, next_paragraph: ParagraphFeatures
    ) -> bool:
        if main_paragraph not in self._alignment_scores:
            return False

        if previous_paragraph.get_distance(next_paragraph) > 0:
            return False

        merged_paragraph = previous_paragraph.model_copy().merge(next_paragraph)
        previous_score = self._alignment_scores[main_paragraph].score
        match_score = ParagraphMatchScore.from_paragraphs_features(main_paragraph, merged_paragraph)
        return previous_score <= match_score.overall_score
