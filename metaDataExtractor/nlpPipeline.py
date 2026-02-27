from textinput import TextInput
from preprocessor import Preprocessor
from intentClassifier import IntentClassifier
from entityExtractor import EntityExtractor
from command import Command


class NLPPipeline:
    """
    Orchestrates the full NLP flow:
      TextInput -> Preprocessor -> IntentClassifier -> EntityExtractor -> Command
    """

    def __init__(self, use_llm: bool = True):
        self.preprocessor = Preprocessor()
        self.classifier = IntentClassifier(use_llm=use_llm)
        self.extractor = EntityExtractor(use_llm=use_llm)

    def process(self, input: TextInput) -> Command:
        # Step 1 — clean the raw text
        cleaned = self.preprocessor.clean(input.raw_text)

        # Step 2 — classify intent
        intent = self.classifier.predict(cleaned)

        # Step 3 — extract entities based on intent
        entities = self.extractor.extract(cleaned, intent)

        return Command(intent=intent, entities=entities)