#!/usr/bin/env python3
import sys

sys.path.insert(0, "src")

from trainable_entity_extractor.domain.FormatSegmentText import FormatSegmentText
from rapidfuzz import fuzz

# Test the failing cases
print("=== Testing fuzzy_match_with_typo ===")
formatter = FormatSegmentText(["This is itme in the text"], "item")
result = formatter.format()
print(f"Result: {result}")
print(f"Expected: contains <b>itme</b>")
print(f'Actual contains: {"<b>itme</b>" in result}')

print("\n=== Testing fuzzy_match_with_punctuation ===")
formatter2 = FormatSegmentText(["This is item_1 in the text"], "item-1")
result2 = formatter2.format()
print(f"Result: {result2}")
print(f"Expected: contains <b>item_1</b>")
print(f'Actual contains: {"<b>item_1</b>" in result2}')

# Debug the fuzzy scores
print("\n=== Fuzzy score debugging ===")
print(f'fuzz.ratio("item", "itme"): {fuzz.ratio("item", "itme")}')
print(f'fuzz.ratio("item-1", "item_1"): {fuzz.ratio("item-1", "item_1")}')
