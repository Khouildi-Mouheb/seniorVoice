import re


class Preprocessor:
    """
    Cleans raw transcribed text from elderly Tunisian speakers.
    Handles:
      - Disfluencies: euh, hm, ah, ...
      - Repeated words: demain demain -> demain
      - Darija datetime normalization -> French equivalents
      - Extra whitespace / punctuation cleanup
    """

    DISFLUENCIES = [
        r"\beuh+\b",
        r"\bhm+\b",
        r"\bah+\b",
        r"\boh+\b",
        r"\bben\b",
        r"\bvoilà\b",
        r"\.{2,}",   # ascii ellipses
        "\u2026",    # unicode ellipsis character …  # ellipses
        r"\s*,\s*,",  # double commas
    ]

    # Darija / Tunisian Arabic -> normalized French equivalents
    DARIJA_DATETIME = {
        # Time of day
        r"\bel sbeh\b":        "le matin",
        r"\bles\s?beh\b":      "le matin",
        r"\bel\s?sbeh\b":      "le matin",
        r"\bes\s?sbeh\b":      "le matin",
        r"\bfi sbeh\b":        "le matin",
        r"\bbaad\s?dhhor\b":   "l'après-midi",
        r"\bb[ae][39]d\s?dh?h?or\b": "l'après-midi",
        r"\bel\s?lil\b":       "le soir",
        r"\bfi\s?llil\b":      "le soir",
        r"\bel\s?asa\b":       "le soir",
        # Days
        r"\bghoudwa\b":        "demain",
        r"\bgh[ou]+dwa\b":     "demain",
        r"\bylo[uo]m\b":       "aujourd'hui",
        r"\bel\s?youm\b":      "aujourd'hui",
        r"\blyoum\b":          "aujourd'hui",
        r"\bwel\s?ghana\b":    "après-demain",
        # Relative time
        r"\bb[ae][39]d\s?s[ae][39]a\b": "dans une heure",
        r"\bb[ae][39]d\s?(\d+)\s?s[ae][39]at\b": r"dans \1 heures",
    }

    def clean(self, text: str) -> str:
        if not text:
            return ""

        result = text.lower().strip()

        # Normalize darija datetime expressions first
        for pattern, replacement in self.DARIJA_DATETIME.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # Remove disfluencies
        for pattern in self.DISFLUENCIES:
            result = re.sub(pattern, " ", result, flags=re.IGNORECASE)

        # Remove repeated consecutive words  e.g. "demain demain" -> "demain"
        result = re.sub(r"\b(\w+)\s+\1\b", r"\1", result)

        # Normalize whitespace
        result = re.sub(r"\s+", " ", result).strip()

        return result