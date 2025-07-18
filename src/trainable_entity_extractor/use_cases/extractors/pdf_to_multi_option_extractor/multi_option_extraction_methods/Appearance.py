from pydantic import BaseModel

from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.Value import Value


class Appearance(BaseModel):
    option_label: str
    context: str

    def to_value(self, option_labels: list[str], options: list[Option]) -> Value:
        option = options[option_labels.index(self.option_label)]
        return Value(id=option.id, label=option.label, segment_text=self.context)

    def __eq__(self, other):
        if not isinstance(other, Appearance):
            return False
        return self.option_label == other.option_label
