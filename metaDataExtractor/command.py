import json
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Command:
    """Final structured output of the NLP pipeline."""
    intent: str
    entities: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {"intent": self.intent, "entities": self.entities},
            ensure_ascii=False,
            indent=2
        )

    def __repr__(self):
        return f"Command(intent={self.intent!r}, entities={self.entities})"