from html import escape
from re import IGNORECASE, compile, Match
from re import escape as re_escape
from rapidfuzz import fuzz
from typing import Optional, List, Tuple


class FormatSegmentText:
    def __init__(self, texts: List[str], label: str = "") -> None:
        self.texts = texts
        self.label = (label or "").strip()

    def format(self) -> str:
        if not self.texts or not any(self.texts):
            return ""
        if self._has_existing_html():
            return "".join(self.texts)
        if not self.label:
            return self._format_all_unlabeled()

        match_indices = self._get_match_indices()
        if not match_indices:
            return self._format_all_unlabeled()

        context_indices = self._get_context_indices(match_indices)
        return self._build_output_from_indices(context_indices, match_indices)

    def _has_existing_html(self) -> bool:
        return any("<p>" in text or "<b>" in text for text in self.texts)

    def _format_all_unlabeled(self) -> str:
        return "".join(f"<p>{escape(text)}</p>" for text in self.texts)

    def _get_match_indices(self) -> List[int]:
        if self._is_date_format():
            return self._get_date_match_indices()
        return self._get_label_match_indices()

    def _get_date_match_indices(self) -> List[int]:
        date_parts = self._get_date_parts()
        if not date_parts:
            return []
        year, month_name, day_variants = date_parts
        components = [year, month_name] + day_variants
        indices: List[int] = []
        for i, text in enumerate(self.texts):
            if any(compile(r"\b" + re_escape(comp) + r"\b", IGNORECASE).search(text) for comp in components if comp):
                indices.append(i)
        return indices

    def _get_label_match_indices(self) -> List[int]:
        pattern = compile(re_escape(self.label), IGNORECASE)
        indices = []

        for i, text in enumerate(self.texts):
            if not text:
                continue

            # First check for exact matches
            if pattern.search(text):
                indices.append(i)
            else:
                # If no exact match, check for fuzzy match
                fuzzy_match = self._find_fuzzy_match_in_text(text)
                if fuzzy_match:
                    indices.append(i)

        return indices

    def _get_context_indices(self, match_indices: List[int]) -> List[int]:
        context = set(match_indices)
        for i in match_indices:
            if i > 0:
                context.add(i - 1)
            if i < len(self.texts) - 1:
                context.add(i + 1)
        return sorted(list(context))

    def _build_output_from_indices(self, context_indices: List[int], match_indices: List[int]) -> str:
        output: List[str] = []
        is_date = self._is_date_format()
        for i in context_indices:
            text = self.texts[i]
            highlighted = self._highlight_text(text, is_date) if i in match_indices else escape(text)
            output.append(f"<p>{highlighted}</p>")
        return "".join(output)

    def _highlight_text(self, text: str, is_date: bool) -> str:
        if is_date:
            date_parts = self._get_date_parts()
            if date_parts:
                return self._highlight_date_in_text(text, *date_parts)
        return self._highlight_label_in_text(text)

    def _highlight_label_in_text(self, text: str) -> str:
        exact_matches = list(compile(re_escape(self.label), IGNORECASE).finditer(text))
        if exact_matches:
            return self._apply_exact_highlights(text, exact_matches)
        fuzzy_match = self._find_fuzzy_match_in_text(text)
        if fuzzy_match:
            return self._apply_fuzzy_highlight(text, fuzzy_match)
        return escape(text)

    @staticmethod
    def _apply_exact_highlights(text: str, matches: List[Match[str]]) -> str:
        parts: List[str] = []
        last = 0
        for match in matches:
            if match.start() > last:
                parts.append(escape(text[last : match.start()]))
            parts.append(f"<b>{escape(match.group(0))}</b>")
            last = match.end()
        parts.append(escape(text[last:]))
        return "".join(parts)

    def _find_fuzzy_match_in_text(self, text: str) -> Optional[Tuple[int, int, str]]:
        best_match: Optional[Tuple[int, int, str]] = None
        best_score = 0
        label_len = len(self.label)
        min_len = max(1, label_len - 2)
        max_len = min(len(text), label_len + 5)  # Increased range for punctuation variations

        # Import re at the top of the method
        import re

        # First, try to find word-like sequences that might contain our target
        # This pattern captures sequences of word characters, hyphens, underscores, and periods
        word_pattern = re.compile(r"\b[\w\-_.]+\b")
        words = [(m.start(), m.end(), m.group()) for m in word_pattern.finditer(text)]

        # Check individual words first (most common case for punctuation differences and typos)
        for start, end, word in words:
            if min_len <= len(word) <= max_len:
                # Calculate fuzzy score
                score = fuzz.ratio(self.label.lower(), word.lower())
                if score >= 75 and score > best_score:  # Lowered threshold to catch more fuzzy matches
                    best_score = score
                    best_match = (start, end, word)

        # If no good word match found, fall back to sliding window approach
        if not best_match:
            for start in range(len(text)):
                for length in range(min_len, min(max_len + 1, len(text) - start + 1)):
                    substring = text[start : start + length]
                    score = fuzz.ratio(self.label.lower(), substring.lower())
                    if score >= 75 and score > best_score:  # Lowered threshold
                        best_score = score
                        best_match = (start, start + length, substring)

        return best_match

    def _is_date_format(self) -> bool:
        return self._get_date_parts() is not None

    def _get_date_parts(self) -> Optional[Tuple[str, str, List[str]]]:
        parts = self.label.split("/")
        if len(parts) != 3:
            return None
        year, month, day = parts
        if not (
            year.isdigit()
            and len(year) == 4
            and month.isdigit()
            and 1 <= int(month) <= 12
            and day.isdigit()
            and 1 <= int(day) <= 31
        ):
            return None
        month_name = self._get_month_name(int(month))
        day_int = int(day)
        day_variants = [str(day_int), day.zfill(2)] if day_int < 10 else [day]
        return year, month_name, day_variants

    @staticmethod
    def _highlight_date_in_text(text: str, year: str, month_name: str, day_variants: List[str]) -> str:
        components = [year, month_name] + day_variants
        all_matches: List[Match[str]] = []

        for component in components:
            if component:
                pattern = compile(r"\b" + re_escape(component) + r"\b", IGNORECASE)
                all_matches.extend(pattern.finditer(text))

        all_matches.sort(key=lambda m: m.start())

        return FormatSegmentText._apply_exact_highlights(text, all_matches)

    @staticmethod
    def _get_month_name(month_num: int) -> str:
        return {
            1: "January",
            2: "February",
            3: "March",
            4: "April",
            5: "May",
            6: "June",
            7: "July",
            8: "August",
            9: "September",
            10: "October",
            11: "November",
            12: "December",
        }.get(month_num, "")

    @staticmethod
    def _apply_fuzzy_highlight(text: str, fuzzy_match: Tuple[int, int, str]) -> str:
        start, end, substring = fuzzy_match
        return f"{escape(text[:start])}<b>{escape(substring)}</b>{escape(text[end:])}"
