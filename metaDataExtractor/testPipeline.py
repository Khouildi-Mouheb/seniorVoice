"""
Real unit tests for the NLP pipeline using Python's unittest framework.
Run with:  python -m unittest test_nlp_pipeline -v
"""
import sys
import os


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nlpPipeline import NLPPipeline
from textinput import TextInput

test_phrases = [
        "نشرب علاجي الساعة 6 المساء",
        "عيط لمرتي",
        "شنوة الطقس اليوم",
        "فكرني ناخذ الدواء بكرة الصباح",
        "ما نقدرش جاني ألم في صدري"]


for f in test_phrases:
    textInput=TextInput(raw_text=f)
    pipe=NLPPipeline()
    command=pipe.process(textInput)
    print(command)



