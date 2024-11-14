from trainable_entity_extractor.extractors.pdf_to_text_extractor.methods.GlinerFirstDateMethod import GlinerFirstDateMethod


class GlinerLastDateMethod(GlinerFirstDateMethod):

    @staticmethod
    def loop_segments(segments):
        for segment in reversed(segments):
            yield segment
