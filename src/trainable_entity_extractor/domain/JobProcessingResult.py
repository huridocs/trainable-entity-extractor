from pydantic import BaseModel


class JobProcessingResult(BaseModel):
    finished: bool
    success: bool
    error_message: str = ""
    gpu_needed: bool = False
