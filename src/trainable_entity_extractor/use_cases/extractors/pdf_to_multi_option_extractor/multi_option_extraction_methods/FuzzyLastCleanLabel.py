import math
from typing import Optional

from rapidfuzz import fuzz

from trainable_entity_extractor.domain.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.Appearance import (
    Appearance,
)
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyFirstCleanLabel import (
    FuzzyFirstCleanLabel,
)


class FuzzyLastCleanLabel(FuzzyFirstCleanLabel):
    def get_appearance(self, pdf_segments: list[PdfDataSegment], options: list[str]) -> Optional[Appearance]:
        for pdf_segment in reversed(pdf_segments):
            for ratio_threshold in range(100, 95, -1):
                for option in options:
                    if len(pdf_segment.text_content) < math.ceil(len(option) * ratio_threshold / 100):
                        continue
                    text = self.remove_accents(pdf_segment.text_content.lower())
                    if fuzz.partial_ratio(option, text) >= ratio_threshold:
                        pdf_segment.ml_label = 1
                        return Appearance(option_label=option, context=pdf_segment.text_content)

        return None
