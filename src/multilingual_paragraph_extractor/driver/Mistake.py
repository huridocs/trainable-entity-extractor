from pydantic import BaseModel
from rapidfuzz import fuzz

from multilingual_paragraph_extractor.domain.AlignmentScore import AlignmentScore
from multilingual_paragraph_extractor.driver.Labels import Labels, LanguagesTexts


class Mistake(BaseModel):
    mistake_number: int
    truth_labels: Labels
    alignment_score: AlignmentScore

    def get_title(self) -> str:
        file = self.truth_labels.main_xml_name.rsplit(".", 1)[0]
        title = self.add_color(f"Mistake n{self.mistake_number}\n", "94")
        title += f"{file}_{self.truth_labels.other_language} Page {self.alignment_score.main_paragraph.page_number}\n"
        title += f"Score {round(self.alignment_score.score * 100, 2)}%"
        return title

    def get_difference(self, text_1: str, text_2: str) -> str:
        if text_1 == text_2:
            return text_1[:75] + "..." + self.add_color(" ok", "92")

        correct_text = ""
        longer_text = text_1 if len(text_1) > len(text_2) else text_2
        shorter_text = text_1 if longer_text == text_2 else text_2
        idx = 0
        for char_1, char_2 in zip(longer_text, shorter_text):
            if char_1 == char_2:
                correct_text += char_1
                idx += 1
                continue
            break
        wrong_text = longer_text[idx:]
        return self.add_break_lines(correct_text + self.add_color(wrong_text, "91"))

    def __str__(self):
        languages_texts = self.get_label()
        print_text = f"\n{self.get_title()}\n\n"

        if languages_texts:
            print_text += f"Main : "
            print_text += self.get_difference(
                self.alignment_score.main_paragraph.original_text, languages_texts.main_language
            )
            print_text += "\n"
            print_text += f"Other: "
            print_text += self.get_difference(
                self.alignment_score.other_paragraph.original_text, languages_texts.other_language
            )
        else:
            print_text += "No label found\n\n"

        return print_text

    @staticmethod
    def add_color(text: str, color: str) -> str:
        return f"\033[{color}m{text}\033[0m"

    @staticmethod
    def add_break_lines(text: str, words_per_line: int = 15) -> str:
        words = text.split()
        lines = [" ".join(words[i : i + words_per_line]) for i in range(0, len(words), words_per_line)]
        return "\n".join(lines)

    def get_label(self) -> LanguagesTexts:
        best_ratio = 0
        best_paragraph = None

        for truth_paragraph in self.truth_labels.paragraphs:
            fuzz_ratio_main = round(
                fuzz.ratio(self.alignment_score.main_paragraph.original_text, truth_paragraph.main_language)
            )
            if fuzz_ratio_main > best_ratio:
                best_ratio = fuzz_ratio_main
                best_paragraph = truth_paragraph

        return best_paragraph
