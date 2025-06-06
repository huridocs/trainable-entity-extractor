from pathlib import Path
import random

from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.Gemini.GeminiRun import (
    GeminiRun,
    CODE_FILE_NAME,
)


class GeminiRunMultiOption(GeminiRun):
    options: list[str] = []
    multi_value: bool = True

    def _set_prompt(self):
        prompt_parts = ["**Examples**\n"]
        for sample_index, sample in enumerate(self.training_samples):
            prompt_parts.append(f"**Example {sample_index + 1}**\n")
            prompt_parts.append("Input:\n")
            prompt_parts.append(f"```{sample.input_text}```\n\n")
            prompt_parts.append("Output (choose one or more from the allowed options):\n")
            prompt_parts.append(f"```{sample.output}```\n\n")

        examples_string = "".join(prompt_parts)
        options_string = ", ".join([f'"{opt}"' for opt in self.options])

        self.prompt = f"""**Task**
        We have a set of example inputs and the corresponding outputs. Each output is a list of one or more options, chosen from a fixed set. Your goal is to infer the logic from the examples and write a self-contained Python function that, given a new input, returns a list of options (strings) from the allowed set that apply.

        Allowed options: [{options_string}]

        {examples_string}

        **Requirements**
        1. Write your solution as a single Python function named `extract(text: str)`. It should return a list of strings, each string being one of the allowed options.
        2. Only return options from the allowed set.
        3. If no options apply, return an empty list.
        4. Only return your function definition, wrapped in a Python code block.
        {'5. Pick only one option at most' if self.multi_value else ''}

        **Output Format**
        Return the function definition *only*, wrapped in fenced code blocks using Python syntax. For example:

        ```python
        def extract(text: str):
            # Your logic here
        ```"""

    def _update_data_from_previous_run(self, previous_run: "GeminiRunMultiOption" = None):
        if previous_run and not previous_run.training_samples:
            selected = []
            for option in self.options:
                for sample in previous_run.mistakes_samples:
                    if isinstance(sample.output, list) and option in sample.output:
                        selected.append(sample)
                        break

            remaining = [s for s in previous_run.mistakes_samples if s not in selected]
            slots = max(0, self.max_training_size - len(selected))
            random.seed(42)
            selected += random.sample(remaining, min(slots, len(remaining))) if slots > 0 else []
            self.training_samples = selected
            self.non_used_samples = [s for s in previous_run.mistakes_samples if s not in selected]
            self._set_prompt()
        else:
            super()._update_data_from_previous_run(previous_run)

    def _update_mistakes_samples(self):
        predictions = self.run_code(self.non_used_samples)
        self.mistakes_samples = [
            sample for sample, prediction in zip(self.non_used_samples, predictions) if set(prediction) != set(sample.output)
        ]

    def _process_samples_with_function(self, extract_func: callable, samples: list) -> list:
        outputs = []
        for sample in samples:
            try:
                result = extract_func(sample.input_text)
                outputs.append(result)
            except Exception:
                outputs.append([])
        return outputs

    @staticmethod
    def from_extractor_identifier_multioption(
        extraction_identifier: ExtractionIdentifier, options: list[Option], multi_value: bool = True
    ) -> "GeminiRunMultiOption":
        path = Path(extraction_identifier.get_path()) / CODE_FILE_NAME
        code = path.read_text(encoding="utf-8") if path.exists() else ""
        return GeminiRunMultiOption(code=code, options=[option.label for option in options], multi_value=multi_value)

    @staticmethod
    def _get_empty_results(samples: list) -> list[list[str]]:
        return [[] for _ in samples]
