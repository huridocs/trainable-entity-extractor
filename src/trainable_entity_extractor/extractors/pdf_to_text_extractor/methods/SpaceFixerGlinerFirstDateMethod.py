import re

from trainable_entity_extractor.data.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.extractors.pdf_to_text_extractor.methods.FirstDateMethod import FirstDateMethod
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.GlinerDateParserMethod import (
    GlinerDateParserMethod,
)


class SpaceFixerGlinerFirstDateMethod(FirstDateMethod):
    @staticmethod
    def contains_year(text: str):
        year_pattern = re.compile(r"(1[0-9]{3}|20[0-9]{2})")
        return bool(year_pattern.search(text.replace(" ", "")))

    def get_date_from_segments(self, segments: list[PdfDataSegment], languages: list[str]) -> str:
        for segment in self.loop_segments(segments):
            if not self.contains_year(segment.text_content):
                continue
            date = GlinerDateParserMethod.get_date([segment.text_content])
            if date:
                segment.ml_label = 1
                return date.strftime("%Y-%m-%d")

        return ""
