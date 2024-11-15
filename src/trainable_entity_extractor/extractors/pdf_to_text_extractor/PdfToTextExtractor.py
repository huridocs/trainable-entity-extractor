from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.TrainingSample import TrainingSample
from trainable_entity_extractor.extractors.ExtractorBase import ExtractorBase
from trainable_entity_extractor.extractors.ToTextExtractor import ToTextExtractor
from trainable_entity_extractor.extractors.ToTextExtractorMethod import ToTextExtractorMethod

from trainable_entity_extractor.extractors.pdf_to_text_extractor.methods.FirstDateMethod import FirstDateMethod
from trainable_entity_extractor.extractors.pdf_to_text_extractor.methods.GlinerFirstDateMethod import GlinerFirstDateMethod
from trainable_entity_extractor.extractors.pdf_to_text_extractor.methods.GlinerLastDateMethod import GlinerLastDateMethod
from trainable_entity_extractor.extractors.pdf_to_text_extractor.methods.LastDateMethod import LastDateMethod
from trainable_entity_extractor.extractors.pdf_to_text_extractor.methods.PdfToTextFastSegmentSelector import (
    PdfToTextFastSegmentSelector,
)
from trainable_entity_extractor.extractors.pdf_to_text_extractor.methods.PdfToTextRegexMethod import PdfToTextRegexMethod
from trainable_entity_extractor.extractors.pdf_to_text_extractor.methods.PdfToTextSegmentSelector import (
    PdfToTextSegmentSelector,
)

from trainable_entity_extractor.extractors.pdf_to_text_extractor.methods.pdf_to_text_method_builder import (
    pdf_to_text_method_builder,
    text_to_text_methods,
)
from trainable_entity_extractor.extractors.segment_selector.FastAndPositionsSegmentSelector import (
    FastAndPositionsSegmentSelector,
)
from trainable_entity_extractor.extractors.segment_selector.FastSegmentSelector import FastSegmentSelector
from trainable_entity_extractor.extractors.segment_selector.SegmentSelector import SegmentSelector
from trainable_entity_extractor.extractors.text_to_text_extractor.methods.MT5TrueCaseEnglishSpanishMethod import (
    MT5TrueCaseEnglishSpanishMethod,
)


class PdfToTextExtractor(ToTextExtractor):
    stand_alone_methods = [
        PdfToTextRegexMethod,
        FirstDateMethod,
        LastDateMethod,
        GlinerFirstDateMethod,
        GlinerLastDateMethod,
    ]

    fast_segment_selector_methods = [
        pdf_to_text_method_builder(PdfToTextFastSegmentSelector, x) for x in text_to_text_methods
    ]
    segment_selector_methods = [pdf_to_text_method_builder(PdfToTextSegmentSelector, x) for x in text_to_text_methods]
    t5_methods = [
        pdf_to_text_method_builder(PdfToTextFastSegmentSelector, MT5TrueCaseEnglishSpanishMethod),
        pdf_to_text_method_builder(PdfToTextSegmentSelector, MT5TrueCaseEnglishSpanishMethod),
    ]

    METHODS: list[type[ToTextExtractorMethod]] = (
        stand_alone_methods + fast_segment_selector_methods + segment_selector_methods + t5_methods
    )

    def create_model(self, extraction_data: ExtractionData) -> tuple[bool, str]:
        if not extraction_data or not extraction_data.samples:
            return False, "No data to create model"

        SegmentSelector(extraction_identifier=self.extraction_identifier).prepare_model_folder()
        FastSegmentSelector(extraction_identifier=self.extraction_identifier).prepare_model_folder()
        FastAndPositionsSegmentSelector(extraction_identifier=self.extraction_identifier).prepare_model_folder()
        return super().create_model(extraction_data)

    @staticmethod
    def get_train_test_sets(extraction_data: ExtractionData) -> (ExtractionData, ExtractionData):
        samples_with_label_segments_boxes = [x for x in extraction_data.samples if x.labeled_data.label_segments_boxes]

        if len(samples_with_label_segments_boxes) < 2 and len(extraction_data.samples) > 10:
            return PdfToTextExtractor.split_80_20(extraction_data)

        if len(samples_with_label_segments_boxes) < 10:
            test_extraction_data = ExtractorBase.get_extraction_data_from_samples(
                extraction_data, samples_with_label_segments_boxes
            )
            return extraction_data, test_extraction_data

        samples_without_label_segments_boxes = [
            x for x in extraction_data.samples if not x.labeled_data.label_segments_boxes
        ]

        train_size = int(len(samples_with_label_segments_boxes) * 0.8)
        train_set: list[TrainingSample] = (
            samples_with_label_segments_boxes[:train_size] + samples_without_label_segments_boxes
        )

        if len(extraction_data.samples) < 15:
            test_set: list[TrainingSample] = samples_with_label_segments_boxes[-10:]
        else:
            test_set = samples_with_label_segments_boxes[train_size:]

        train_extraction_data = ExtractorBase.get_extraction_data_from_samples(extraction_data, train_set)
        test_extraction_data = ExtractorBase.get_extraction_data_from_samples(extraction_data, test_set)
        return train_extraction_data, test_extraction_data

    @staticmethod
    def split_80_20(extraction_data):
        train_size = int(len(extraction_data.samples) * 0.8)
        train_set: list[TrainingSample] = extraction_data.samples[:train_size]
        test_set = extraction_data.samples[train_size:]
        train_extraction_data = ExtractorBase.get_extraction_data_from_samples(extraction_data, train_set)
        test_extraction_data = ExtractorBase.get_extraction_data_from_samples(extraction_data, test_set)
        return train_extraction_data, test_extraction_data

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        for sample in extraction_data.samples:
            if sample.pdf_data and sample.pdf_data.contains_text():
                return True

        return False
