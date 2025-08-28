from html import escape
from re import IGNORECASE, compile
from re import escape as re_escape
from rapidfuzz import fuzz


class FormatSegmentText:

    def __init__(self, texts: list[str], label: str = ""):
        self.texts = texts
        self.label = (label or "").strip()
        self.combined_text = " ".join(text for text in texts if text)

    def format(self) -> str:
        if not self.texts or not any(self.texts):
            return ""

        if not self.label:
            formatted_texts = [f"<p>{escape(text)}</p>" for text in self.texts if text]
            return "".join(formatted_texts)

        best_match = self._find_best_match()
        if best_match:
            if best_match["type"] == "exact":
                formatted = self._highlight_exact_matches(best_match["matches"])
            else:
                formatted = self._highlight_fuzzy_match(best_match["match"])
        else:
            formatted = escape(self.combined_text)

        return f"<p>{formatted}</p>"

    def _find_best_match(self):
        fuzzy_match = self._find_best_fuzzy_match()
        exact_matches = self._find_exact_matches()

        if exact_matches and fuzzy_match:
            fuzzy_start, fuzzy_end, fuzzy_substring = fuzzy_match
            fuzzy_score = fuzz.ratio(self.label.lower(), fuzzy_substring.lower())
            fuzzy_length = fuzzy_end - fuzzy_start
            exact_length = sum(match.end() - match.start() for match in exact_matches)

            if (fuzzy_score >= 90 and fuzzy_length >= exact_length) or (fuzzy_length > exact_length and fuzzy_score >= 85):
                return {"type": "fuzzy", "match": fuzzy_match}

            return {"type": "exact", "matches": exact_matches}

        if fuzzy_match:
            return {"type": "fuzzy", "match": fuzzy_match}
        if exact_matches:
            return {"type": "exact", "matches": exact_matches}

        return None

    def _find_exact_matches(self):
        pattern = compile(re_escape(self.label), IGNORECASE)
        return list(pattern.finditer(self.combined_text))

    def _find_best_fuzzy_match(self):
        best_match = None
        best_score = 0
        label_len = len(self.label)
        min_len = max(1, label_len - 2)
        max_len = min(len(self.combined_text), label_len + 3)

        for start in range(len(self.combined_text)):
            for length in range(min_len, min(max_len + 1, len(self.combined_text) - start + 1)):
                substring = self.combined_text[start : start + length]
                score = fuzz.ratio(self.label.lower(), substring.lower())

                if score >= 80 and score > best_score:
                    best_score = score
                    best_match = (start, start + length, substring)

        return best_match

    def _highlight_exact_matches(self, matches):
        parts = []
        last = 0
        for match in matches:
            if match.start() > last:
                parts.append(escape(self.combined_text[last : match.start()]))
            parts.append(f"<b>{escape(match.group(0))}</b>")
            last = match.end()
        parts.append(escape(self.combined_text[last:]))
        return "".join(parts)

    def _highlight_fuzzy_match(self, match):
        start, end, substring = match
        escaped_before = escape(self.combined_text[:start])
        escaped_match = escape(substring)
        escaped_after = escape(self.combined_text[end:])
        return f"{escaped_before}<b>{escaped_match}</b>{escaped_after}"
