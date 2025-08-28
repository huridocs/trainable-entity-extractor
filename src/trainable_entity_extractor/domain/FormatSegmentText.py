from html import escape
from re import IGNORECASE, compile, Pattern, Match
from re import escape as re_escape
from rapidfuzz import fuzz
from typing import Optional, List, Tuple


class FormatSegmentText:

    def __init__(self, texts: List[str], label: str = "") -> None:
        self.texts = texts
        self.label = (label or "").strip()
        self.combined_text = " ".join(text for text in texts if text)

    def format(self) -> str:
        if not self.texts or not any(self.texts):
            return ""

        if self._has_existing_html():
            return "".join(self.texts)

        if not self.label:
            return self._format_without_label()

        if self._is_date_format():
            return self._format_date() or self._format_without_label()

        return self._format_with_label()

    def _has_existing_html(self) -> bool:
        return "<b>" in self.combined_text or "<p>" in self.combined_text

    def _format_without_label(self) -> str:
        return "".join(f"<p>{escape(text)}</p>" for text in self.texts if text)

    def _format_date(self) -> Optional[str]:
        date_formatted = self._highlight_date_components()
        return date_formatted if date_formatted is not None else None

    def _format_with_label(self) -> str:
        texts_to_process = self._select_context_texts()
        highlighted_texts: List[str] = []

        for text in texts_to_process:
            if not text:
                continue

            highlighted_text = self._highlight_text(text)
            highlighted_texts.append(f"<p>{highlighted_text}</p>")

        return "".join(highlighted_texts) if highlighted_texts else self._format_without_label()

    def _select_context_texts(self) -> List[str]:
        if not self.label or not self.texts:
            return self.texts

        pattern = compile(re_escape(self.label), IGNORECASE)
        match_indices = [i for i, t in enumerate(self.texts) if t and pattern.search(t)]

        if not match_indices:
            return self.texts

        context_indices = set()

        for idx in match_indices:
            context_indices.add(idx)
            if idx - 1 >= 0:
                context_indices.add(idx - 1)
            if idx + 1 < len(self.texts):
                context_indices.add(idx + 1)

        ordered = [self.texts[i] for i in sorted(context_indices)]

        return ordered

    def _highlight_text(self, text: str) -> str:
        exact_matches = self._find_exact_matches_in_text(text)
        if exact_matches:
            return self._apply_exact_highlights(text, exact_matches)

        fuzzy_match = self._find_fuzzy_match_in_text(text)
        if fuzzy_match:
            return self._apply_fuzzy_highlight(text, fuzzy_match)

        return escape(text)

    def _find_exact_matches_in_text(self, text: str) -> List[Match[str]]:
        pattern = compile(re_escape(self.label), IGNORECASE)
        return list(pattern.finditer(text))

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
        max_len = min(len(text), label_len + 3)
        for start in range(len(text)):
            for length in range(min_len, min(max_len + 1, len(text) - start + 1)):
                substring = text[start : start + length]
                score = fuzz.ratio(self.label.lower(), substring.lower())
                if score >= 80 and score > best_score:
                    best_score = score
                    best_match = (start, start + length, substring)
        return best_match

    @staticmethod
    def _apply_fuzzy_highlight(text: str, fuzzy_match: Tuple[int, int, str]) -> str:
        start, end, substring = fuzzy_match
        return f"{escape(text[:start])}<b>{escape(substring)}</b>{escape(text[end:])}"

    def _is_date_format(self) -> bool:
        parts = self.label.split("/")
        if len(parts) != 3:
            return False
        year, month, day = parts
        return (
            year.isdigit()
            and len(year) == 4
            and month.isdigit()
            and 1 <= int(month) <= 12
            and day.isdigit()
            and 1 <= int(day) <= 31
        )

    def _highlight_date_components(self) -> Optional[str]:
        parts = self.label.split("/")
        year, month_num, day = parts
        month_name = self._get_month_name(int(month_num))
        highlighted_texts: List[str] = []
        overall_highlighted = False
        for text in self.texts:
            current_text = self._highlight_date_in_text(text, year, month_name, day)
            if "<b>" in current_text:
                overall_highlighted = True
            highlighted_texts.append(
                f"<p>{escape(current_text).replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')}</p>"
            )
        return "".join(highlighted_texts) if overall_highlighted else None

    @staticmethod
    def _highlight_date_in_text(text: str, year: str, month_name: str, day: str) -> str:
        current_text = text
        components = [year, month_name]
        day_int = int(day)
        day_components = [str(day_int), day.zfill(2)] if day_int < 10 else [day]
        components.extend(day_components)
        for component in components:
            if component and component in current_text:
                pattern = compile(r"\b" + re_escape(component) + r"\b", IGNORECASE)
                if pattern.search(current_text):
                    current_text = pattern.sub(f"<b>{component}</b>", current_text)

        return current_text

    @staticmethod
    def _get_month_name(month_num: int) -> str:
        months = {
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
        }
        return months.get(month_num, "")
