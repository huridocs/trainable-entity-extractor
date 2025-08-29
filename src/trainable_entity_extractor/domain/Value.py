from pydantic import BaseModel

from trainable_entity_extractor.domain.Option import Option


class Value(BaseModel):
    id: str
    label: str
    segment_text: str = ""
    __hash__ = object.__hash__

    @staticmethod
    def from_option(option: Option) -> "Value":
        return Value(id=option.id, label=option.label)

    def __eq__(self, other):
        if not isinstance(other, Value):
            return False

        if other.segment_text and self.segment_text != other.segment_text:
            return False

        return self.id == other.id and self.label == other.label
