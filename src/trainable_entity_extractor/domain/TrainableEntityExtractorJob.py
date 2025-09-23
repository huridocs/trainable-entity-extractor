from pydantic import BaseModel

from trainable_entity_extractor.domain.Option import Option


class TrainableEntityExtractorJob(BaseModel):
    run_name: str
    extraction_name: str
    extractor_name: str
    method_name: str
    multi_label: bool = False
    options: list[Option] = []
    gpu_needed: bool
    timeout: int
    should_be_retrained_with_more_data: bool = False
