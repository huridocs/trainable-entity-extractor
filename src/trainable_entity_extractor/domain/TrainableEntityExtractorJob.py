from pydantic import BaseModel

from trainable_entity_extractor.domain.Option import Option


class TrainableEntityExtractorJob(BaseModel):
    run_name: str
    extraction_name: str
    extractor_name: str
    method_name: str
    multi_value: bool = False
    options: list[Option] = []
    gpu_needed: bool
    timeout: int
    output_path: str = ""

    def set_extractors_path(self, path: str) -> "TrainableEntityExtractorJob":
        self.output_path = path
        return self
