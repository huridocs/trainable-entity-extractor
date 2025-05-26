from google import genai

from trainable_entity_extractor.config import GEMINI_API_KEY
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.use_cases.extractors.ToTextExtractorMethod import ToTextExtractorMethod


class GeminiFlashTextMethod(ToTextExtractorMethod):

    client = genai.Client(api_key=GEMINI_API_KEY)

    def train(self, extraction_data: ExtractionData):
        pass

    def predict(self, predictions_samples: list[PredictionSample]) -> list[str]:
        pass

    @staticmethod
    def execute_function(code: str, function_name: str = None, *args, **kwargs):
        namespace = {}
        try:
            exec(code, namespace)

            if function_name is None:
                import re

                match = re.search(r"def\s+(\w+)\s*\(", code)
                if match:
                    function_name = match.group(1)
                else:
                    return {"success": False, "result": None, "error": "Could not find function name in code"}

            func = namespace[function_name]
            result = func(*args, **kwargs)
            if not result:
                result = None
            return {"success": True, "result": result, "error": None}

        except Exception as e:
            return {"success": False, "result": None, "error": str(e)}

    @staticmethod
    def get_model_answer(examples: list[LabeledDataSample], model: str) -> str:
        client = Client()
        prompt = get_prompt(examples)
        response = client.chat(model=model, messages=[{"role": "user", "content": prompt}])
        answer: str = response["message"]["content"]
        code_start = "```python\n"
        code_end = "```"
        code_part = answer[answer.find(code_start) + len(code_start) : answer.rfind(code_end)]
        return code_part

    @staticmethod
    def get_actual_output(code_to_execute: str, data: LabeledData) -> list[str]:
        actual_output = []
        for sample in data.samples:
            function_return = execute_function(code_to_execute, "extract", sample.text)
            if not function_return["success"]:
                actual_output.append("FAIL")
                continue
            actual_output.append(function_return["result"])
        return actual_output

    @staticmethod
    def update_examples(example_indexes, examples, mistake_indexes, train_data) -> bool:
        new_example_added = False
        for i in mistake_indexes:
            if i not in example_indexes:
                examples.append(train_data.samples[i])
                example_indexes.append(i)
                new_example_added = True
                break
        return new_example_added


if __name__ == "__main__":
    print(GeminiFlashTextMethod.train("21 DE MARÇO DE 2023"))
