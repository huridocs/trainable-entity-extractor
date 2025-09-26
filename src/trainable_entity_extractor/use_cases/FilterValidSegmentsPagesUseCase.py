import json
import os
import re
from pathlib import Path

from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionData import PredictionData

MAX_PAGES = 99999


class FilterValidSegmentsPagesUseCase:
    def __init__(self, extraction_identifier: ExtractionIdentifier):
        self.labeled_data_json_path = Path(extraction_identifier.get_path(), "filter_pages.json")
        self.start_gaps = []
        self.end_gaps = []
        self.valid_pages_ranges = []

    def get_valid_pages(self, total_number_pages_per_document: list[int]) -> list[list[int]]:
        if self.valid_pages_ranges == [MAX_PAGES]:
            return self.get_valid_pages_when_no_labeled_data(total_number_pages_per_document)

        if min(self.start_gaps) <= min(self.end_gaps):
            start = min(self.start_gaps)
            end = max(self.start_gaps) + max(self.valid_pages_ranges)
            return [self.get_range(start, end, number_of_pages) for number_of_pages in total_number_pages_per_document]

        valid_page_numbers_from_the_end = []

        for number_of_pages in total_number_pages_per_document:
            start = number_of_pages - max(self.end_gaps) - max(self.valid_pages_ranges)
            end = number_of_pages - min(self.end_gaps)
            valid_page_numbers_from_the_end.append(self.get_range(start, end, number_of_pages))

        return valid_page_numbers_from_the_end

    def for_training(self, labeled_data_list: list[LabeledData]):
        if not self.labeled_data_json_path.parent.exists():
            os.makedirs(self.labeled_data_json_path.parent)

        self.labeled_data_json_path.write_text(json.dumps([x.model_dump_json() for x in labeled_data_list]))
        self.set_parameters(labeled_data_list)
        pages_list = [[x.page_number for x in labeled_data.xml_segments_boxes] for labeled_data in labeled_data_list]
        total_number_pages_per_document = [max(pages) if pages else 1000 for pages in pages_list]
        return self.get_valid_pages(total_number_pages_per_document)

    def for_prediction(self, prediction_data_list: list[PredictionData]):
        try:
            with open(self.labeled_data_json_path, "r") as file:
                labeled_data_dict = json.load(file)
                labeled_data_list = [LabeledData(**json.loads(x)) for x in labeled_data_dict]
                self.set_parameters(labeled_data_list)
        except:
            self.set_parameters([])

        pages_list = [
            [x.page_number for x in prediction_data.xml_segments_boxes] for prediction_data in prediction_data_list
        ]
        number_pages_per_document = [max(pages) if pages else 1000 for pages in pages_list]

        return self.get_valid_pages(number_pages_per_document)

    def set_parameters(self, labeled_data_list):
        if not labeled_data_list or sum([len(x.label_segments_boxes) for x in labeled_data_list]) == 0:
            self.start_gaps = [0]
            self.end_gaps = [0]
            self.valid_pages_ranges = [MAX_PAGES]
            return

        for labeled_data in labeled_data_list:
            if not labeled_data.label_segments_boxes:
                continue

            pages = [x.page_number for x in labeled_data.xml_segments_boxes]
            number_of_pages = max(pages) if pages else 1000
            valid_page_numbers = [x.page_number for x in labeled_data.label_segments_boxes]
            self.valid_pages_ranges.append(max(valid_page_numbers) - min(valid_page_numbers) + 1)
            self.start_gaps.append(min(valid_page_numbers) - 1)
            self.end_gaps.append(number_of_pages - max(valid_page_numbers))

        if not self.valid_pages_ranges:
            self.start_gaps = [0]
            self.end_gaps = [0]
            self.valid_pages_ranges = [0]

    @staticmethod
    def get_range(start: int, end: int, number_pages: int) -> list[int]:
        start = max(0, start - 1)
        end = min(end + 1, number_pages)
        return list(range(start + 1, end + 1))

    @staticmethod
    def correct_page_numbers(xml_content: str, page_numbers: list[str], ends_of_pages: list[int]):
        if not page_numbers and not ends_of_pages:
            return True

        if len(ends_of_pages) != len(page_numbers):
            return False

        if len(page_numbers) == 1 and page_numbers[0].isnumeric():
            return True

        for page_number, next_page_number in zip(page_numbers, page_numbers[1:]):
            if not page_number.isnumeric() or not next_page_number.isnumeric():
                return False

            if int(page_number) > int(next_page_number):
                return False

        for page_number in page_numbers:
            page_tag = re.search(f'<page number="{page_number}"', xml_content)
            valid_ends = [end_of_page for end_of_page in ends_of_pages if end_of_page > page_tag.start()]

            if not valid_ends:
                return False

        return True

    @staticmethod
    def filter_xml_pages(xml_content: str, page_numbers_to_keep: list[int]):
        if not page_numbers_to_keep:
            return xml_content

        pages_xml = re.findall('<page number="[0-9]*"', xml_content)
        page_numbers = [id_field.replace('"', "").replace("<page number=", "") for id_field in pages_xml]
        ends_of_pages = [x.end() for x in re.finditer("</page>", xml_content)]

        if not FilterValidSegmentsPagesUseCase.correct_page_numbers(xml_content, page_numbers, ends_of_pages):
            return xml_content

        pages_to_remove = [page_number for page_number in page_numbers if int(page_number) not in page_numbers_to_keep]

        for page_number in reversed(pages_to_remove):
            page_tag = re.search(f'<page number="{page_number}"', xml_content)
            ends = [end_of_page for end_of_page in ends_of_pages if end_of_page > page_tag.start()]
            removed_texts = re.sub("<text.*?</text>", "", xml_content[page_tag.start() : ends[0]])
            xml_content = "".join([xml_content[: page_tag.start()], removed_texts, xml_content[ends[0] :]])

        return xml_content

    @staticmethod
    def get_valid_pages_when_no_labeled_data(total_number_pages_per_document: list[int]) -> list[list[int]]:
        valid_page_numbers = []
        for number_of_pages in total_number_pages_per_document:
            valid_page_number = list(range(1, min(5, number_of_pages + 1)))
            valid_page_number += list(range(number_of_pages - 3, number_of_pages + 1))
            valid_page_number = list(set([x for x in valid_page_number if x > 0]))
            valid_page_numbers.append(valid_page_number)

        return valid_page_numbers
