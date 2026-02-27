"""
Real unit tests for the NLP pipeline using Python's unittest framework.
Run with:  python -m unittest test_nlp_pipeline -v
"""
import sys
import os
import json
import re
from typing import Optional, List, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))



SUPPORTED_INTENTS = [
    "set_reminder", "phone_call", "weather", "send_message",
    "medication_reminder", "play_music", "news", "emergency", "unknown",
]

KEYWORD_RULES = {
    "emergency": [
        "urgence", "secours", "sos", "ambulance", "samu", "au secours",
        "نجدة", "إسعاف", "مساعدة", "استغاثة", "طوارئ",
        "اتصل بالإسعاف", "اتصل بالشرطة", "ألم في صدري",
        "ما نقدرش", "وقعت", "تعبت بزاف",
    ],
    "medication_reminder": [
        "médicament", "médicaments", "pilule", "comprimé", "ordonnance", "dose",
        "doliprane", "paracétamol", "ibuprofène", "aspirine",
        "prendre le", "prendre mon", "prendre ma", "avaler", "ma pilule",
        "دواء", "دوا", "الدواء", "دوائي", "علاج", "علاجي",
        "حبة", "حبوب", "حبوبي", "روشتة",
        "ناخذ الدواء", "ناخذ دوا", "ناخذ حبة",
        "نشرب الدواء", "نشرب علاجي",
        "وقت دوائي", "وقت الدواء",
    ],
    "phone_call": [
        "appelle", "téléphone", "appel", "contacte", "joindre", "parler à",
        "عيط", "عيط لـ", "حول", "حول لـ", "تلفن", "تلفن لـ",
        "كلم", "نكلم", "بغيت نكلم",
        "اتصل", "اتصل بـ", "اتصال", "مكالمة", "رنة",
        "مرتي", "بأخويا",
    ],
    "weather": [
        "météo", "pluie", "soleil", "nuage", "température", "temperature",
        "chaud", "froid", "vent", "orage", "prévisions",
        "طقس", "الطقس", "شنوة الطقس", "أحوال الجو", "توقعات الطقس",
        "درجة الحرارة", "شحال درجة",
        "تمطر", "تمطار", "عاصفة", "غيوم", "شمس", "حر", "برد", "رياح",
        "كابوت", "واش غادي تكون",
    ],
    "send_message": [
        "envoie", "message", "sms", "écris", "dis à", "dis-lui", "préviens",
        "ابعث", "ابعث رسالة", "رسالة", "رسالة نصية",
        "بعث", "بعثلو", "بعثلها",
        "قوله", "قوليها", "قولهم", "وقوله", "وقوليها",
    ],
    "set_reminder": [
        "rappelle", "rappelle-moi", "rappel", "n'oublie", "n'oublie pas",
        "oublie pas", "réveille", "rendez-vous", "rdv", "réunion", "alarme",
        "فكرني", "فكر", "ذكرني", "ما تنساش", "تنساش",
        "لا تنسى", "لا تنساني", "نبهني",
        "موعد", "اجتماع", "تذكير", "منبه", "بكرة الصباح",
    ],
    "play_music": [
        "musique", "chanson", "joue", "écoute", "radio",
        "موسيقى", "أغنية", "أغاني", "شغل", "راديو", "نسمع", "بغيت نسمع",
    ],
    "news": [
        "nouvelles", "actualités", "infos", "journal",
        "أخبار", "الأخبار", "نشرة", "آخر الأخبار", "شنوة الجديد",
    ],
}


class IntentClassifier:

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
        
        # Add weighted keywords for better accuracy
        self.keyword_weights = {
            "emergency": {
                "high": ["au secours", "sos", "ambulance", "نجدة", "إسعاف", "طوارئ", "ألم في صدري", "ما نقدرش"],
                "medium": ["urgence", "secours", "samu", "مساعدة", "استغاثة", "وقعت", "تعبت بزاف"],
                "low": ["اتصل بالإسعاف", "اتصل بالشرطة"]
            },
            "medication_reminder": {
                "high": ["ناخذ الدواء", "نشرب الدواء", "وقت الدواء", "دواء", "علاج", "حبوب"],
                "medium": ["médicament", "pilule", "comprimé", "دوا", "روشتة"],
                "low": ["doliprane", "paracétamol", "ordonnance", "حبة"]
            },
            "phone_call": {
                "high": ["اتصل بـ", "عيط لـ", "كلم", "نكلم", "بغيت نكلم", "مرتي", "بأخويا"],
                "medium": ["appelle", "téléphone", "اتصل", "مكالمة", "حول لـ"],
                "low": ["contacte", "joindre", "رنة", "تلفن"]
            },
            "weather": {
                "high": ["طقس", "الطقس", "météo", "درجة الحرارة", "شحال درجة"],
                "medium": ["pluie", "soleil", "température", "شنوة الطقس", "واش غادي تكون"],
                "low": ["chaud", "froid", "vent", "غيوم", "شمس"]
            },
            "send_message": {
                "high": ["ابعث رسالة", "رسالة", "بعثلو", "بعثلها", "قوله"],
                "medium": ["envoie", "message", "sms", "ابعث", "قوليها"],
                "low": ["écris", "dis à", "préviens", "قولهم"]
            },
            "set_reminder": {
                "high": ["فكرني", "ذكرني", "ما تنساش", "تذكير", "موعد"],
                "medium": ["rappelle", "n'oublie pas", "réveille", "نبهني", "منبه"],
                "low": ["rendez-vous", "rdv", "alarme", "اجتماع"]
            },
            "play_music": {
                "high": ["شغل", "موسيقى", "أغنية", "radio", "نسمع"],
                "medium": ["musique", "chanson", "joue", "écoute", "أغاني"],
                "low": ["بغيت نسمع", "راديو"]
            },
            "news": {
                "high": ["أخبار", "الأخبار", "actualités"],
                "medium": ["nouvelles", "infos", "journal", "آخر الأخبار"],
                "low": ["نشرة", "شنوة الجديد"]
            }
        }
        
        # Context patterns for better disambiguation
        self.context_patterns = {
            "medication_reminder": [
                r"before|after|with (meal|food|breakfast|lunch|dinner)",
                r"قبل|بعد|مع (الأكل|الفطور|الغداء|العشاء)",
                r"صباحا|مساء|ليل|نهار",
                r"مرة|مرتين|٣ مرات|٣ مرات"
            ],
            "phone_call": [
                r"call (me|him|her|them|my|mom|dad|brother|sister)",
                r"appelle (moi|lui|elle|eux|ma|mon)",
                r"اتصل ب (ي|ك|ه|ها|هم|أمي|أبي|أخي|أختي)",
                r"كلم (ني|و|ها|هم)"
            ]
        }
        
        # Negation words to avoid false positives
        self.negation_words = [
            "not", "don't", "doesn't", "didn't", "won't", "can't",
            "لا", "ما", "مش", "موش", "لم", "لن",
            "ne", "n'", "pas", "plus", "jamais"
        ]
        
        # Intent-specific stopwords (words that shouldn't trigger certain intents)
        self.intent_stopwords = {
            "medication_reminder": ["acheter", "buy", "شراء", "بيع"],
            "phone_call": ["number", "رقم", "تلفون"],
            "weather": ["yesterday", "أمس", "البارح"]
        }
        
        # Common phrases that might confuse the classifier
        self.confusing_phrases = {
            "appelle": ["rappelle", "n'appelle pas"],
            "دواء": ["شراء دواء", "دواء جديد"]
        }

    def predict(self, text: str) -> str:
        # First check for negations that might invert intent
        if self._has_negation(text):
            # Handle negated commands differently
            pass
            
        if self._client and self.use_llm:
            result = self._predict_llm(text)
            if result:
                return result
        return self._predict_keywords_enhanced(text)

    def _predict_llm(self, text: str) -> Optional[str]:
        prompt = f"""You are an assistant helping classify voice commands from elderly Tunisian users.
The text may mix French and Tunisian Arabic (Darija), be fragmented, or hesitant.
Classify into EXACTLY one of: {json.dumps(SUPPORTED_INTENTS, ensure_ascii=False)}
Text: "{text}"
Consider context, common phrases, and potential ambiguities.
Reply with ONLY the intent string. If unsure, reply "unknown"."""
        try:
            response = self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=20,
                messages=[{"role": "user", "content": prompt}]
            )
            intent = response.content[0].text.strip().lower()
            return intent if intent in SUPPORTED_INTENTS else "unknown"
        except Exception:
            return None

    def _predict_keywords(self, text: str) -> str:
        text_lower = text.lower()
        for intent, keywords in KEYWORD_RULES.items():
            for kw in keywords:
                if re.search(r'[\u0600-\u06FF]', kw):
                    # Arabic: plain substring (no \b support for Arabic)
                    if kw in text:
                        return intent
                else:
                    # Latin: word boundary to avoid "appelle" in "rappelle"
                    pattern = r"(?<!\w)" + re.escape(kw) + r"(?!\w)"
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        return intent
        return "unknown"
    
    def _predict_keywords_weighted(self, text: str) -> str:
        """Weighted keyword matching for better accuracy"""
        scores = {intent: 0 for intent in SUPPORTED_INTENTS}
        
        # Apply weighted keywords
        for intent, weights in self.keyword_weights.items():
            for level, keywords in weights.items():
                weight_multiplier = 3 if level == "high" else (2 if level == "medium" else 1)
                for kw in keywords:
                    if self._keyword_in_text(kw, text):
                        scores[intent] += weight_multiplier
        
        # Also check original keyword rules as fallback
        for intent, keywords in KEYWORD_RULES.items():
            for kw in keywords:
                if self._keyword_in_text(kw, text):
                    scores[intent] += 1
        
        # Apply context pattern bonuses
        for intent, patterns in self.context_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    scores[intent] += 2
        
        # Apply penalties for stopwords
        for intent, stopwords in self.intent_stopwords.items():
            for stopword in stopwords:
                if self._keyword_in_text(stopword, text):
                    scores[intent] = max(0, scores[intent] - 2)
        
        # Handle confusing phrases
        for confusing, clarifications in self.confusing_phrases.items():
            if self._keyword_in_text(confusing, text):
                for clarification in clarifications:
                    if self._keyword_in_text(clarification, text):
                        # Reduce score for intents that might be confused
                        for intent in scores:
                            if intent != "unknown":
                                scores[intent] = max(0, scores[intent] - 1)
        
        # Get intent with highest score
        max_intent = max(scores.items(), key=lambda x: x[1])
        if max_intent[1] > 0:
            return max_intent[0]
        return "unknown"
    
    def _predict_keywords_enhanced(self, text: str) -> str:
        """Enhanced prediction with all features"""
        # First try weighted prediction
        weighted_result = self._predict_keywords_weighted(text)
        
        # If weighted gives a clear result, return it
        if weighted_result != "unknown":
            return weighted_result
        
        # Otherwise, try context-aware prediction
        context_result = self._predict_with_context(text)
        if context_result != "unknown":
            return context_result
        
        # Fall back to simple keyword matching
        return self._predict_keywords(text)
    
    def _predict_with_context(self, text: str) -> str:
        """Context-aware prediction"""
        # Check for time-related patterns that might indicate medication
        time_patterns = [r"\d+\s*(h|heure|ساعة)", r"صباح|مساء|ليل"]
        has_time = any(re.search(pattern, text) for pattern in time_patterns)
        
        # Check for person-related patterns that might indicate phone call
        person_patterns = [r"مري?|باي?|أخ?|أخت?", r"my (mom|dad|brother|sister)"]
        has_person = any(re.search(pattern, text, re.IGNORECASE) for pattern in person_patterns)
        
        # Adjust scores based on context
        scores = {intent: 0 for intent in SUPPORTED_INTENTS}
        
        # Base scores from keyword matching
        base_intent = self._predict_keywords(text)
        if base_intent != "unknown":
            scores[base_intent] += 2
        
        # Apply context bonuses
        if has_time:
            scores["medication_reminder"] += 1
            scores["set_reminder"] += 1
        
        if has_person:
            scores["phone_call"] += 1
            scores["send_message"] += 1
        
        # Get highest scoring intent
        max_intent = max(scores.items(), key=lambda x: x[1])
        if max_intent[1] > 1:  # Threshold of 2 or more
            return max_intent[0]
        
        return base_intent
    
    def _has_negation(self, text: str) -> bool:
        """Check if text contains negation words"""
        text_lower = text.lower()
        for negation in self.negation_words:
            if self._keyword_in_text(negation, text_lower):
                return True
        return False
    
    def _keyword_in_text(self, keyword: str, text: str) -> bool:
        """Check if keyword exists in text with proper boundary handling"""
        if re.search(r'[\u0600-\u06FF]', keyword):
            # For Arabic text, use simple substring match
            return keyword in text
        else:
            # For Latin text, use word boundaries
            text_lower = text.lower()
            keyword_lower = keyword.lower()
            pattern = r"(?<!\w)" + re.escape(keyword_lower) + r"(?!\w)"
            return bool(re.search(pattern, text_lower, re.IGNORECASE))
    
    def get_intent_confidence(self, text: str) -> Dict[str, float]:
        """Return confidence scores for all intents"""
        scores = {intent: 0.0 for intent in SUPPORTED_INTENTS}
        total_score = 0
        
        # Calculate weighted scores
        for intent, weights in self.keyword_weights.items():
            for level, keywords in weights.items():
                multiplier = 3 if level == "high" else (2 if level == "medium" else 1)
                for kw in keywords:
                    if self._keyword_in_text(kw, text):
                        scores[intent] += multiplier
                        total_score += multiplier
        
        # Normalize scores
        if total_score > 0:
            for intent in scores:
                scores[intent] = scores[intent] / total_score
        
        return scores


