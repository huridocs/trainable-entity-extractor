import re
from trainable_entity_extractor.adapters.extractors.GlinerDateExtractor import GlinerDateExtractor
from trainable_entity_extractor.adapters.extractors.ToTextExtractorMethod import ToTextExtractorMethod
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PdfDataSegment import PdfDataSegment
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.GlinerDateParserMethod import (
    GlinerDateParserMethod,
)
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData


class GlinerFirstDateMethod(ToTextExtractorMethod):
    def train(self, extraction_data: ExtractionData):
        languages = [x.labeled_data.language_iso for x in extraction_data.samples]
        self.save_json("languages.json", list(set(languages)))

    def predict(self, prediction_samples_data: PredictionSamplesData) -> list[str]:
        gliner_model = GlinerDateExtractor.get_model()
        predictions_samples = prediction_samples_data.prediction_samples
        predictions = [""] * len(predictions_samples)
        languages = self.load_json("languages.json")
        for index, prediction_sample in enumerate(predictions_samples):
            segments = prediction_sample.pdf_data.pdf_data_segments

            if predictions[index] or not prediction_sample.pdf_data or not segments:
                continue

            predictions[index] = self.get_date_from_segments(gliner_model, segments, languages)

        return predictions

    @staticmethod
    def loop_segments(segments: list[PdfDataSegment]):
        for segment in segments:
            yield segment

    @staticmethod
    def contains_year(text: str):
        year_pattern = re.compile(r"([0-9]{2})")
        return bool(year_pattern.search(text.replace(" ", "")))

    def get_date_from_segments(self, model, segments: list[PdfDataSegment], languages):
        merge_segments: list[list[PdfDataSegment]] = self.merge_segments_for_dates(segments)
        for segments in merge_segments:
            segment_merged = PdfDataSegment.from_list_to_merge(segments)
            if not self.contains_year(segment_merged.text_content):
                continue

            date = GlinerDateParserMethod.get_date(model, [segment_merged.text_content])
            if date:
                for segment in segments:
                    segment.ml_label = 1
                return date.strftime("%Y-%m-%d")

        return ""

    def merge_segments_for_dates(self, segments: list[PdfDataSegment]):
        min_words = 35
        merge_segments: list[list[PdfDataSegment]] = list()
        for segment in segments:
            if not merge_segments:
                merge_segments.append([segment])
                continue

            words_previous_segment = self.count_segments_words(merge_segments[-1])

            if words_previous_segment < min_words:
                merge_segments[-1].append(segment)

            merge_segments.append([segment])

        return merge_segments

    @staticmethod
    def count_segments_words(segments: list[PdfDataSegment]):
        return sum([len(segment.text_content.split()) for segment in segments])
