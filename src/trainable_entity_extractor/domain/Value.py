from pydantic import BaseModel
from trainable_entity_extractor.domain.Option import Option
from html import escape
from re import IGNORECASE, compile
from re import escape as re_escape
from rapidfuzz import fuzz


class Value(BaseModel):
    id: str
    label: str
    segment_text: str = ""
    __hash__ = object.__hash__

    def __init__(self, **data):
        super().__init__(**data)
        self._format_segment_text()

    def _format_segment_text(self):
        if not self.segment_text:
            return

        if '<p>' in self.segment_text or '<br' in self.segment_text or '<b>' in self.segment_text:
            return

        text = self.segment_text
        label = (self.label or '').strip()

        if not label:
            self.segment_text = f"<p>{escape(text)}</p>"
            return

        best_match = self._find_best_match(text, label)
        if best_match:
            if best_match['type'] == 'exact':
                formatted = self._highlight_exact_matches(text, best_match['matches'])
            else:
                formatted = self._highlight_fuzzy_match(text, best_match['match'])
        else:
            formatted = escape(text)

        self.segment_text = f"<p>{formatted}</p>"

    @staticmethod
    def _find_best_match(text, label):
        fuzzy_match = Value._find_best_fuzzy_match(text, label)
        exact_matches = Value._find_exact_matches(text, label)

        if exact_matches and fuzzy_match:
            fuzzy_start, fuzzy_end, fuzzy_substring = fuzzy_match
            fuzzy_score = fuzz.ratio(label.lower(), fuzzy_substring.lower())
            fuzzy_length = fuzzy_end - fuzzy_start
            exact_length = sum(match.end() - match.start() for match in exact_matches)

            if (fuzzy_score >= 90 and fuzzy_length >= exact_length) or \
               (fuzzy_length > exact_length and fuzzy_score >= 85):
                return {'type': 'fuzzy', 'match': fuzzy_match}

            return {'type': 'exact', 'matches': exact_matches}

        if fuzzy_match:
            return {'type': 'fuzzy', 'match': fuzzy_match}
        if exact_matches:
            return {'type': 'exact', 'matches': exact_matches}

        return None

    @staticmethod
    def _find_exact_matches(text, label):
        pattern = compile(re_escape(label), IGNORECASE)
        return list(pattern.finditer(text))

    @staticmethod
    def _find_best_fuzzy_match(text, label):
        best_match = None
        best_score = 0
        label_len = len(label)
        min_len = max(1, label_len - 2)
        max_len = min(len(text), label_len + 3)

        for start in range(len(text)):
            for length in range(min_len, min(max_len + 1, len(text) - start + 1)):
                substring = text[start:start + length]
                score = fuzz.ratio(label.lower(), substring.lower())

                if score >= 80 and score > best_score:
                    best_score = score
                    best_match = (start, start + length, substring)

        return best_match

    @staticmethod
    def _highlight_exact_matches(text, matches):
        parts = []
        last = 0
        for match in matches:
            if match.start() > last:
                parts.append(escape(text[last:match.start()]))
            parts.append(f"<b>{escape(match.group(0))}</b>")
            last = match.end()
        parts.append(escape(text[last:]))
        return ''.join(parts)

    @staticmethod
    def _highlight_fuzzy_match(text, match):
        start, end, substring = match
        escaped_before = escape(text[:start])
        escaped_match = escape(substring)
        escaped_after = escape(text[end:])
        return f"{escaped_before}<b>{escaped_match}</b>{escaped_after}"

    @staticmethod
    def from_option(option: Option) -> "Value":
        return Value(id=option.id, label=option.label)

    def __eq__(self, other):
        if not isinstance(other, Value):
            return False

        if other.segment_text and self.segment_text != other.segment_text:
            return False

        return self.id == other.id and self.label == other.label
