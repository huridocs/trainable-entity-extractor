from pydantic import BaseModel


class ExtractionDistributedTask(BaseModel):
    run_name: str
    extraction_name: str
    extractor_name: str
    method_name: str
    gpu_needed: bool
    timeout: int
