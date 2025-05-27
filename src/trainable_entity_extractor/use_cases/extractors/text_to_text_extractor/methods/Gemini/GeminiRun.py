import codecs
import random
from pathlib import Path

from google import genai
from pydantic import BaseModel

from trainable_entity_extractor.config import GEMINI_API_KEY
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.Gemini.GeminiSample import \
    GeminiSample

CODE_FILE_NAME = "gemini_code.py"


class GeminiRun(BaseModel):
    max_training_size: int = 0
    training_samples: list[GeminiSample] = list()
    non_used_samples: list[GeminiSample] = list()
    mistakes_samples: list[GeminiSample] = list()
    prompt: str = ""
    code: str = ""

    def set_run_data(self, previous_run: "GeminiRun" = None):
        if not previous_run:
            return

        random.seed(42)
        samples_previous_run = previous_run.training_samples
        max_samples_to_add = self.max_training_size - len(samples_previous_run)
        max_samples_to_add = min(max_samples_to_add, len(previous_run.mistakes_samples))
        self.training_samples = samples_previous_run + random.sample(previous_run.mistakes_samples, max_samples_to_add)

        training_samples_set = set(self.training_samples)
        self.non_used_samples = [
            sample for sample in previous_run.mistakes_samples if sample not in training_samples_set
        ]
        self.set_prompt()

    def run_training(self, previous_run: "GeminiRun"):
        if not self.max_training_size or not GEMINI_API_KEY:
            return
        
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        self.set_run_data(previous_run)
        self.set_code(client)
        self.set_mistakes_samples()

    def set_mistakes_samples(self):
        predictions = self.run_code(self.non_used_samples)
        self.mistakes_samples = [
            sample for sample, prediction in zip(self.non_used_samples, predictions)
            if prediction.strip() != sample.output_text.strip()
        ]

    def set_code(self, client):
        response = client.models.generate_content(model="gemini-2.5-flash-preview-05-20", contents=self.prompt)
        answer: str = response.text
        code_start = "```python\n"
        code_end = "```"
        self.code = answer[answer.find(code_start) + len(code_start): answer.rfind(code_end)]

    def set_prompt(self):
        prompt_parts = ["**Examples**\n"]
        for sample_index, sample in enumerate(self.training_samples):
            prompt_parts.append(f"**Example {sample_index + 1}**\n")
            prompt_parts.append("Input:\n")
            prompt_parts.append(f'"{sample.input_text}"\n\n')
            prompt_parts.append("Output:\n")
            prompt_parts.append(f"{sample.output_text}\n\n")

        examples_string = "".join(prompt_parts)

        self.prompt = f"""**Task**
                       We have a set of example inputs and the corresponding outputs. These examples illustrate how we want to transform the input data to the output data. Your goal is to figure out the pattern or logic from these examples and write a self-contained Python function that reproduces this behavior.
    
                       We do not provide an explicit list of rules. Instead, use the examples to infer how the input should be processed to create the output. If the pattern does not clearly match some new input, your function may return `None` or an empty string, but it should handle the provided examples correctly.
    
                       {examples_string}
    
                       **Requirements**
                       1. Write your solution as a single Python function named `extract(text: str)`. No additional arguments should be required.
                          2. In your solution, you may apply any logic needed to extract, parse, or process the input so that it matches how each example is transformed to its output.
                          3. If no valid transformation or pattern is found, return `None` or an empty string.
                          4. Only return your function definition. No additional commentary, test calls, or example usage should appear outside the code block.
                          5. Your code should be standalone and use only standard Python 3 libraries without external dependencies.
    
    
                       **Output Format**
                       Return the function definition *only*, wrapped in fenced code blocks using Python syntax. For example:
    
                       ```python
                       def extract(text: str):
                           # Your logic here
                       ```
    
                       No explanatory text or test calls should appear outside of that code block."""

    def save_code(self, extraction_identifier: ExtractionIdentifier):
        if not self.code:
            return

        try:
            code_to_save = codecs.decode(self.code.encode('utf-8', 'backslashreplace'), 'unicode-escape')
        except Exception:
            code_to_save = self.code.replace('\\n', '\n')

        extraction_identifier.save_content(CODE_FILE_NAME, code_to_save)

    @staticmethod
    def _get_empty_results(samples: list[GeminiSample]) -> list[str]:
            return [""] * len(samples)

    def _load_extract_function(self):
        local_namespace = {}
        if self.code.startswith('"') and self.code.endswith("'"):
            self.code = self.code[1:-1]
        elif self.code.startswith("'") and self.code.endswith("'"):
            self.code = self.code[1:-1]
        elif self.code.startswith('"') and self.code.endswith('"'):
            self.code = self.code[1:-1]

        try:
            # Ensure self.code is treated as a string that might contain Python-style escapes
            code_to_execute = codecs.decode(self.code.encode('utf-8', 'backslashreplace'), 'unicode-escape')
        except Exception:
            # Fallback if the above decoding fails for some reason
            code_to_execute = self.code.replace('\\n', '\n')

        try:
            exec(code_to_execute, {}, local_namespace)
            extract_func = local_namespace.get("extract")
            if callable(extract_func):
                return extract_func
            return None
        except Exception:
            return None

    @staticmethod
    def _process_samples_with_function(extract_func: callable, samples: list[GeminiSample]) -> list[str]:
        outputs_texts = []
        for sample in samples:
            try:
                result = extract_func(sample.input_text)
                outputs_texts.append(str(result) if result is not None else "")
            except Exception:
                outputs_texts.append("")
        return outputs_texts

    def run_code(self, samples: list[GeminiSample]) -> list[str]:
        if not self.code:
            return self._get_empty_results(samples)

        extract_func = self._load_extract_function()
        if not extract_func:
            return self._get_empty_results(samples)

        return self._process_samples_with_function(extract_func, samples)
        
    @staticmethod
    def from_extractor_identifier(extraction_identifier: ExtractionIdentifier) -> "GeminiRun":
        path = Path(extraction_identifier.get_path()) / CODE_FILE_NAME
        code = path.read_text(encoding="utf-8") if path.exists() else ""
        return GeminiRun(code=code)