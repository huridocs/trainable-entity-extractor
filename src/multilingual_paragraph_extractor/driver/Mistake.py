import difflib

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

    def get_difference(self, label: str, prediction: str) -> str:
        if label == prediction:
            return self.add_color("ok", "92") + " " + self.add_color(label[:70] + "...", "90")

        m = difflib.SequenceMatcher(a=label, b=prediction)

        result = "\nLabel        :  "
        result += self.add_color(self.add_break_lines(label), "90")
        result += "\n\nPrediction   :  "
        differences = ""
        for tag, i1, i2, j1, j2 in m.get_opcodes():
            if tag == "replace":
                differences += self.add_color(label[i1:i2], "92")
                differences += self.add_color(prediction[j1:j2], "91")
            if tag == "delete":
                differences += self.add_color(label[i1:i2], "92")
            if tag == "insert":
                differences += self.add_color(prediction[j1:j2], "91")
            if tag == "equal":
                differences += f"{label[i1:i2]}"
        result += f"{self.add_break_lines(differences)}"
        return result

    def __str__(self):
        languages_texts = self.get_label()
        print_text = f"\n{self.get_title()}\n\n"

        if languages_texts:
            print_text += f"Main language : "
            print_text += self.get_difference(
                languages_texts.main_language,
                self.alignment_score.main_paragraph.original_text,
            )
            print_text += "\n\n"
            print_text += f"Other language: "
            print_text += self.get_difference(
                languages_texts.other_language, self.alignment_score.other_paragraph.original_text
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
