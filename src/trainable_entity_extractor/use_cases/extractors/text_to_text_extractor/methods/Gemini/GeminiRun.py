import random
from pathlib import Path
import builtins

from google import genai
from pydantic import BaseModel
import textwrap

from trainable_entity_extractor.config import GEMINI_API_KEY
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.Gemini.GeminiSample import GeminiSample


CODE_FILE_NAME = "gemini_code.py"
PROMPT_FILE_NAME = "prompt.txt"


class GeminiRun(BaseModel):
    gemini_model: str = "gemini-2.5-flash"
    max_training_size: int = 0
    prompt: str = ""
    code: str = ""
    training_samples: list[GeminiSample] = list()
    non_used_samples: list[GeminiSample] = list()
    mistakes_samples: list[GeminiSample] = list()
    from_class_name: str = ""

    def _update_data_from_previous_run(self, previous_run: "GeminiRun" = None):
        if not previous_run:
            return

        random.seed(42)
        samples_previous_run = previous_run.training_samples
        max_samples_to_add = self.max_training_size - len(samples_previous_run)
        max_samples_to_add = min(max_samples_to_add, len(previous_run.mistakes_samples))

        if max_samples_to_add > 0 and len(previous_run.mistakes_samples) > 0:
            self.training_samples = samples_previous_run + random.sample(previous_run.mistakes_samples, max_samples_to_add)
        else:
            self.training_samples = samples_previous_run

        training_samples_set = set(self.training_samples)
        self.non_used_samples = [sample for sample in previous_run.mistakes_samples if sample not in training_samples_set]
        self._set_prompt()

    def run_training(self, previous_run: "GeminiRun"):
        if not self.max_training_size or not GEMINI_API_KEY:
            return

        self._update_data_from_previous_run(previous_run)

        if len(self.training_samples) == len(previous_run.training_samples):
            self.mistakes_samples = previous_run.mistakes_samples
            self.code = previous_run.code
            return

        self._set_code_from_model()
        self._update_mistakes_samples()

    def _update_mistakes_samples(self):
        predictions = self.run_code(self.non_used_samples)
        self.mistakes_samples = [
            sample
            for sample, prediction in zip(self.non_used_samples, predictions)
            if prediction.strip() != sample.output.strip()
        ]

    def _set_code_from_model(self):
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(model=self.gemini_model, contents=self.prompt)
        answer: str = response.text
        code_start = "```python\n"
        code_end = "```"
        self.code = answer[answer.find(code_start) + len(code_start) : answer.rfind(code_end)]

    def _get_task_section(self, indent_prefix: str) -> str:
        """Override in subclasses to customize the task section"""

        task_raw = f"""We have a set of example inputs and the corresponding outputs. These examples illustrate how we want to transform the input data to the output data. Your goal is to figure out the pattern or logic from these examples and write a self-contained Python function that reproduces this behavior.
    We do not provide an explicit list of rules. Instead, use the examples to infer how the input should be processed to create the output. If the pattern does not clearly match some new input, your function may return `None` or an empty string, but it should handle the provided examples correctly."""
        return textwrap.indent(textwrap.dedent(task_raw), indent_prefix)

    def _get_examples_section(self, indent_prefix: str) -> str:
        """Override in subclasses to customize the examples section"""
        example_blocks = []
        for i, sample in enumerate(self.training_samples, 1):
            block = f"""**Example {i}**
    Input:
    ```{sample.input_text}```
    Output:
    ```{sample.output}```"""
            example_blocks.append(block)
        return textwrap.indent("\n\n".join(example_blocks), indent_prefix)

    def _get_requirements_section(self, indent_prefix: str) -> str:
        """Override in subclasses to customize the requirements section"""
        reqs = [
            "1. Write your solution as a single Python function named `extract(text: str)`. No additional arguments should be required.",
            "2. Only return your function definition. No additional commentary, test calls, or example usage should appear outside the code block.",
            "3. If no valid transformation or pattern is found, return an empty string.",
            "4. Generalize as much as possible based on the examples provided.",
            "5. Your code should be standalone and use only standard Python 3 libraries without external dependencies.",
            "6. Important: Put all the import statements inside the extract function.",
        ]
        return textwrap.indent("\n".join(reqs), indent_prefix)

    def _get_special_requirements_section(self, indent_prefix: str) -> str:
        """Can be overridden in subclasses to customize special requirements"""
        special_reqs_raw = """Select the best extraction method from [Pure Python, Regex, rapidfuzz, LightGBM] based on the nature of the task and examples provided."""
        return textwrap.indent(textwrap.dedent(special_reqs_raw), indent_prefix)

    def _get_output_format_section(self, indent_prefix: str) -> str:
        """Can be overridden in subclasses to customize output format"""
        fmt_raw = """Return the function definition \\*only\\*, wrapped in fenced code blocks using Python syntax. For example:

    ```python
    def extract(text: str):
        # Your logic here
    ```"""
        return textwrap.indent(textwrap.dedent(fmt_raw), indent_prefix)

    def _set_prompt(self):
        indent_prefix = "    "  # 4 spaces for indentation

        # Get all sections
        task = self._get_task_section(indent_prefix)
        examples = self._get_examples_section(indent_prefix)
        requirements = self._get_requirements_section(indent_prefix)
        special_requirements = self._get_special_requirements_section(indent_prefix)
        output_format = self._get_output_format_section(indent_prefix)

        # Assemble final prompt
        self.prompt = (
            f"**Task**\n{task}\n\n"
            f"**Examples**\n{examples}\n\n"
            f"**Requirements**\n{requirements}\n\n"
            f"**Special Requirements**\n{special_requirements}\n\n"
            f"**Output Format**\n{output_format}"
        )

    def get_file_name(self, file_name: str) -> str:
        return str(Path(self.from_class_name) / file_name)

    def save_code(self, extraction_identifier: ExtractionIdentifier):
        if not self.code:
            return

        code_to_save = self.code.replace("\\n", "\n")
        code_to_save = code_to_save.replace("\\t", "\t")
        code_to_save = code_to_save.replace("\\r", "\r")

        extraction_identifier.save_content(self.get_file_name(CODE_FILE_NAME), code_to_save, False)
        extraction_identifier.save_content(self.get_file_name(PROMPT_FILE_NAME), self.prompt, False)

    def _load_extract_function(self):
        import re
        import json
        import math
        import datetime
        import collections
        import itertools
        import string
        import rapidfuzz

        local_namespace = {}

        code_to_execute = self.code.replace("\\n", "\n")
        code_to_execute = code_to_execute.replace("\\t", "\t")
        code_to_execute = code_to_execute.replace("\\r", "\r")

        try:
            global_namespace = {
                "__builtins__": builtins,
                "__name__": "__main__",
                "re": re,
                "json": json,
                "math": math,
                "datetime": datetime,
                "collections": collections,
                "itertools": itertools,
                "string": string,
                "rapidfuzz": rapidfuzz,
            }

            exec(code_to_execute, global_namespace, local_namespace)
            extract_func = local_namespace.get("extract")
            if callable(extract_func):
                return extract_func
            return None
        except Exception as e:
            print(f"Error loading extract function: {e}")
            return None

    def _process_samples_with_function(self, extract_func: callable, samples: list[GeminiSample]) -> list[str]:
        outputs_texts = []
        for sample in samples:
            try:
                result = extract_func(sample.input_text)
                outputs_texts.append(str(result) if result is not None else "")
            except Exception:
                outputs_texts.append("")

        outputs_texts = [self.clean_outputs(text) for text in outputs_texts]
        return outputs_texts

    def run_code(self, samples: list[GeminiSample]) -> list[str] | list[list[str]]:
        if not self.code:
            return self._get_empty_results(samples)

        extract_func = self._load_extract_function()
        if not extract_func:
            return self._get_empty_results(samples)

        return self._process_samples_with_function(extract_func, samples)

    @staticmethod
    def from_extractor_identifier(extraction_identifier: ExtractionIdentifier, from_class_name: str) -> "GeminiRun":
        path = Path(extraction_identifier.get_path()) / from_class_name / CODE_FILE_NAME
        code = path.read_text(encoding="utf-8") if path.exists() else ""
        return GeminiRun(code=code, from_class_name=from_class_name)

    @staticmethod
    def _get_empty_results(samples: list[GeminiSample]) -> list[str]:
        return [""] * len(samples)

    @staticmethod
    def clean_outputs(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        return text
