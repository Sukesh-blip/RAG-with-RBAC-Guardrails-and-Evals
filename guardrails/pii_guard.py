"""
PII detection and redaction using Presidio.
Applied to the final answer before it's returned - this is about minimizing
what the chatbot exposes, separate from RBAC (which controls what documents
are retrievable in the first place). A role being authorized to see a
document doesn't mean every raw identifier in it belongs in a chat answer.
"""

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

_analyzer = AnalyzerEngine()
_anonymizer = AnonymizerEngine()

# Entity types redacted from any chatbot answer, regardless of role.
# Names and figures like salary are left intact since they're often the
# actual thing being asked about - this targets contact/financial
# identifiers that are rarely necessary for a chat response to include.
REDACT_ENTITIES = [
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "CREDIT_CARD",
    "US_BANK_NUMBER",
    "IBAN_CODE",
]


def redact_pii(text: str) -> dict:
    """Scans text for PII and returns a redacted version plus what was found."""
    results = _analyzer.analyze(text=text, entities=REDACT_ENTITIES, language="en")

    if not results:
        return {"redacted_text": text, "entities_found": []}

    anonymized = _anonymizer.anonymize(text=text, analyzer_results=results)
    entities_found = [
        {"type": r.entity_type, "score": round(r.score, 2)} for r in results
    ]

    return {"redacted_text": anonymized.text, "entities_found": entities_found}