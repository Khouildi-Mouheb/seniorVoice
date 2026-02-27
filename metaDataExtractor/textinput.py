class TextInput:
    """Holds the raw transcribed text coming from ASR."""

    def __init__(self, raw_text: str):
        self.raw_text = raw_text

    def __repr__(self):
        return f"TextInput(raw_text={self.raw_text!r})"