"""Language detection and tokenising.

Telugu and Hindi use their own Unicode blocks, so checking which script the
characters belong to is more reliable than a statistical detector on short text.
"""
import re

LANGUAGES = {"en": "English", "hi": "Hindi", "te": "Telugu"}

_TELUGU = re.compile(r"[\u0C00-\u0C7F]")
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
# Split on spaces and punctuation only. (\w would cut Telugu/Hindi words apart
# at vowel signs, which Python does not count as letters.)
_TOKEN = re.compile(r"[^\s.,;:!?()\[\]\"'।\-/]+")


def detect_language(text: str) -> str:
    te = len(_TELUGU.findall(text))
    hi = len(_DEVANAGARI.findall(text))
    if te == 0 and hi == 0:
        return "en"          # includes romanised Hinglish/Tenglish for now
    return "te" if te >= hi else "hi"


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())
