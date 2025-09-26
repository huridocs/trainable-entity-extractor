import dataclasses
import hashlib
from statistics import mode

from trainable_entity_extractor.domain.PdfData import PdfData


@dataclasses.dataclass
class Modes:
    lines_space_mode: float
    left_space_mode: float
    right_space_mode: float
    font_size_mode: float
    font_family_name_mode: str
    font_family_mode: int
    font_family_mode_normalized: float
    pdf_segments: PdfData

    def __init__(self, pdf_segments: PdfData):
        self.pdf_segments = pdf_segments
        self.set_modes()

    def set_modes(self):
        line_spaces, right_spaces, left_spaces = [0], [0], [0]
        font_sizes, font_ids = list(), list()

        if self.pdf_segments.pdf_features:
            for page, token in self.pdf_segments.pdf_features.loop_tokens():
                right_spaces.append(page.page_width - token.bounding_box.right)
                left_spaces.append(token.bounding_box.left)
                line_spaces.append(token.bounding_box.bottom)
                font_sizes.append(token.font.font_size)
                font_ids.append(token.font.font_id)

        self.font_size_mode = mode(font_sizes) if font_sizes else 0
        self.font_family_name_mode = mode(font_ids) if font_ids else ""
        self.font_family_mode = abs(
            int(
                str(hashlib.sha256(self.font_family_name_mode.encode("utf-8")).hexdigest())[:8],
                16,
            )
        )
        self.font_family_mode_normalized = float(f"{str(self.font_family_mode)[0]}.{str(self.font_family_mode)[1:]}")

        self.lines_space_mode = mode(line_spaces)
        self.left_space_mode = mode(left_spaces)
        self.right_space_mode = mode(right_spaces)
