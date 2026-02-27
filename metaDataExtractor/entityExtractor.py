import re
import os
import json
from typing import Optional
from datetime import datetime, timedelta


# Time expression patterns (French + some Darija already normalized by preprocessor)
TIME_PATTERNS = [
    (r"\bà\s?(\d{1,2})[h:]\s?(\d{2})\b",   lambda m: f"{m.group(1).zfill(2)}:{m.group(2)}"),
    (r"\bà\s?(\d{1,2})\s?h(?:eure)?s?\b",   lambda m: f"{m.group(1).zfill(2)}:00"),
    (r"\b(\d{1,2})[h:](\d{2})\b",            lambda m: f"{m.group(1).zfill(2)}:{m.group(2)}"),
    (r"\b(\d{1,2})\s?heures?\b",             lambda m: f"{m.group(1).zfill(2)}:00"),
    (r"\ble matin\b",                         lambda m: "matin"),
    (r"\bl[' ]?après-?midi\b",               lambda m: "après-midi"),
    (r"\ble soir\b",                          lambda m: "soir"),
    (r"\bce soir\b",                          lambda m: "soir"),
]

DATE_PATTERNS = [
    (r"\baujourd[' ]?hui\b",                 "aujourd'hui"),
    (r"\bdemain\b",                           "demain"),
    (r"\baprès-?demain\b",                    "après-demain"),
    (r"\blundi\b",                            "lundi"),
    (r"\bmardi\b",                            "mardi"),
    (r"\bmercredi\b",                         "mercredi"),
    (r"\bjeudi\b",                            "jeudi"),
    (r"\bvendredi\b",                         "vendredi"),
    (r"\bsamedi\b",                           "samedi"),
    (r"\bdimanche\b",                         "dimanche"),
    (r"\b(\d{1,2})[/\-](\d{1,2})(?:[/\-](\d{2,4}))?\b",
     lambda m: f"{m.group(1).zfill(2)}/{m.group(2).zfill(2)}" + (f"/{m.group(3)}" if m.group(3) else "")),
]

# Contact indicators — words that often precede a contact name
CONTACT_PREFIXES = [
    "appelle", "téléphone à", "appel à", "contacte", "parler à",
    "envoie à", "message à", "dis à", "préviens",
    "le docteur", "la docteure", "dr", "dr.",
    "mon fils", "ma fille", "mon frère", "ma sœur", "ma mère", "mon père",
    "mon mari", "ma femme", "mamie", "papi",
]


class EntityExtractor:
    """
    Extracts datetime, contact, and location entities from cleaned text.

    Strategy:
      1. Regex rules for time/date (fast, reliable)
      2. LLM for contact and ambiguous cases if available
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self._client = None

        if use_llm:
            try:
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                pass

    def extract(self, text: str, intent: str) -> dict:
        entities = {}

        # Always try to extract time and date with regex
        time_val = self._extract_time(text)
        date_val = self._extract_date(text)

        if time_val:
            entities["time"] = time_val
        if date_val:
            entities["date"] = date_val

        # Extract contact for call/message intents
        if intent in ("phone_call", "send_message", "set_reminder"):
            contact = self._extract_contact(text, intent)
            if contact:
                entities["contact"] = contact

        # Extract location for weather
        if intent == "weather":
            location = self._extract_location(text)
            if location:
                entities["location"] = location

        # Extract medication info
        if intent == "medication_reminder":
            med = self._extract_medication(text)
            if med:
                entities["medication"] = med

        # Fill missing slots with LLM if available
        if self._client and self.use_llm:
            entities = self._enrich_with_llm(text, intent, entities)

        return entities

    # ------------------------------------------------------------------ #
    #  Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _extract_time(self, text: str) -> Optional[str]:
        for pattern, formatter in TIME_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return formatter(m) if callable(formatter) else formatter
        return None

    def _extract_date(self, text: str) -> Optional[str]:
        for pattern, value in DATE_PATTERNS:
            if callable(value):
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    return value(m)
            else:
                if re.search(pattern, text, re.IGNORECASE):
                    return value
        return None

    def _extract_contact(self, text: str, intent: str) -> Optional[str]:
        text_lower = text.lower()
        for prefix in CONTACT_PREFIXES:
            idx = text_lower.find(prefix)
            if idx != -1:
                after = text[idx + len(prefix):].strip()
                # grab next 1-3 words as the contact name
                words = after.split()
                name_words = []
                for w in words[:3]:
                    clean = re.sub(r"[^\w'-]", "", w)
                    if clean and clean.lower() not in ("à", "le", "la", "les", "de", "du"):
                        name_words.append(clean)
                if name_words:
                    return " ".join(name_words)
        return None

    def _extract_location(self, text: str) -> Optional[str]:
        m = re.search(r"\bà\s+([A-ZÀ-Ü][a-zà-ü]+(?:\s[A-ZÀ-Ü][a-zà-ü]+)?)\b", text)
        if m:
            return m.group(1)
        m = re.search(r"\b(tunis|sfax|sousse|bizerte|nabeul|kairouan|monastir|gabès)\b", text, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()
        return None

    def _extract_medication(self, text: str) -> Optional[str]:
        m = re.search(
            r"\b(doliprane|paracétamol|ibuprofène|aspirine|amoxicilline|metformine|"
            r"amlodipine|oméprazole|ventoline|\w+ine|\w+ol)\b",
            text, re.IGNORECASE
        )
        return m.group(1) if m else None

    def _enrich_with_llm(self, text: str, intent: str, existing: dict) -> dict:
        """Ask the LLM to fill in any missing slots."""
        missing = []
        if intent in ("set_reminder", "medication_reminder") and "time" not in existing:
            missing.append("time (HH:MM or time-of-day)")
        if intent in ("set_reminder", "medication_reminder") and "date" not in existing:
            missing.append("date (relative: aujourd'hui/demain/...)")
        if intent == "phone_call" and "contact" not in existing:
            missing.append("contact (person to call)")
        if intent == "weather" and "location" not in existing:
            missing.append("location (city)")

        if not missing:
            return existing

        prompt = f"""Extract the following from this Tunisian French/Darija voice command:
{json.dumps(missing, ensure_ascii=False)}

Text: "{text}"

Return ONLY a JSON object with the found values. If a value is not present, omit its key.
Example: {{"time": "10:00", "date": "demain"}}"""

        try:
            response = self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            llm_entities = json.loads(raw)
            # Only add keys not already found by regex
            for k, v in llm_entities.items():
                if k not in existing and v:
                    existing[k] = v
        except Exception:
            pass

        return existing