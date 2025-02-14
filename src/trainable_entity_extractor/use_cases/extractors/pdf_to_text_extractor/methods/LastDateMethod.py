from trainable_entity_extractor.use_cases.extractors.pdf_to_text_extractor.methods.FirstDateMethod import FirstDateMethod


class LastDateMethod(FirstDateMethod):

    @staticmethod
    def loop_segments(segments):
        for segment in reversed(segments):
            yield segment
