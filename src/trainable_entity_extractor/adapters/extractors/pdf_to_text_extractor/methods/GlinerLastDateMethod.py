from trainable_entity_extractor.adapters.extractors.pdf_to_text_extractor.methods.GlinerFirstDateMethod import (
    GlinerFirstDateMethod,
)
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.GlinerDateParserMethod import (
    GlinerDateParserMethod,
)


class GlinerLastDateMethod(GlinerFirstDateMethod):

    @staticmethod
    def loop_segments(segments):
        for segment in reversed(segments):
            yield segment

    def get_date_from_segments(self, segments, languages):
        for segment in self.loop_segments(segments):
            if not self.contains_year(segment.text_content):
                continue

            date = GlinerDateParserMethod.get_date([segment.text_content])
            if date:
                segment.ml_label = 1
                return date.strftime("%Y-%m-%d")

        return ""
