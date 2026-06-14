"""
Helper comun pentru parsarea răspunsurilor LLM care conțin JSON.
"""
import re


def extract_json(text: str) -> str:
    """
    Extrage blocul JSON dintr-un răspuns LLM.
    """
    if not text:
        return "{}"

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    brace = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if brace:
        return brace.group(1).strip()

    return text.strip()
