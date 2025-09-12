from pydantic import BaseModel


class TrainableEntityExtractorJob(BaseModel):
    run_name: str
    extraction_name: str
    extractor_name: str
    method_name: str
    gpu_needed: bool
    timeout: int
