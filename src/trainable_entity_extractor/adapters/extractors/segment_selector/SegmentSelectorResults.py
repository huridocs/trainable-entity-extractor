from pydantic import BaseModel


class SegmentSelectorResults(BaseModel):
    method: str
    dataset: str
    samples: int
    precision: float
    recall: float
    seconds: int

    @staticmethod
    def get_padding():
        return {
            "method": "right",
            "dataset": "right",
            "samples": "left",
            "precision": "left",
            "recall": "left",
            "seconds": "left",
        }
