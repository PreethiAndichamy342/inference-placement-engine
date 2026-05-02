"""
PHI De-identification using Microsoft Presidio.

DeIdentifier detects Protected Health Information (PHI) entities in free text
and replaces them with deterministic tokens such as <PERSON_1>, <DATE_1>,
<SSN_1>.  It returns both the anonymised text and an entity_map that maps each
token back to the original value, so the response can be re-identified after
the inference backend returns.

Supported entity types
----------------------
General PII
    PERSON, PHONE_NUMBER, EMAIL_ADDRESS, LOCATION, DATE_TIME,
    US_SSN, US_PASSPORT, US_DRIVER_LICENSE, IBAN_CODE, CREDIT_CARD,
    IP_ADDRESS, URL

Healthcare-specific (custom recognisers)
    MEDICAL_LICENSE    — US medical licence numbers  (pattern)
    NPI_NUMBER         — US National Provider Identifier (10-digit, pattern)
    MRN                — Medical Record Number (pattern heuristic)
    ICD_CODE           — ICD-10 diagnosis codes (pattern)
    NDC_CODE           — National Drug Code (5-4-2 or 5-4 format, pattern)
    HEALTH_PLAN_ID     — Generic health plan / payer ID (pattern)

Usage
-----
    deid = DeIdentifier()
    result = deid.de_identify("Patient John Smith, DOB 01/15/1980, SSN 123-45-6789")
    # result.anonymized_text  → "Patient <PERSON_1>, DOB <DATE_TIME_1>, SSN <US_SSN_1>"
    # result.entity_map       → {"<PERSON_1>": "John Smith", ...}
    # result.entity_count     → 3
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Healthcare-specific custom recognisers
# ---------------------------------------------------------------------------

def _medical_license_recognizer() -> PatternRecognizer:
    """US Medical License numbers: letters + digits, 6-12 chars."""
    return PatternRecognizer(
        supported_entity="MEDICAL_LICENSE",
        patterns=[
            Pattern(
                name="medical_license",
                regex=r"\b[A-Z]{1,3}[-\s]?\d{5,8}\b",
                score=0.6,
            ),
        ],
        context=["license", "lic", "medical license", "state license", "dea"],
    )


def _npi_recognizer() -> PatternRecognizer:
    """US National Provider Identifier — exactly 10 digits."""
    return PatternRecognizer(
        supported_entity="NPI_NUMBER",
        patterns=[
            Pattern(
                name="npi_number",
                regex=r"\bNPI[:\s#]*(\d{10})\b",
                score=0.85,
            ),
            Pattern(
                name="npi_bare",
                regex=r"\b1[0-9]{9}\b",  # NPIs start with 1
                score=0.4,
            ),
        ],
        context=["npi", "national provider", "provider id", "provider number"],
    )


def _mrn_recognizer() -> PatternRecognizer:
    """Medical Record Numbers — typically 6-10 digits, often labelled."""
    return PatternRecognizer(
        supported_entity="MRN",
        patterns=[
            Pattern(
                name="mrn_labelled",
                regex=r"\b(?:MRN|medical record(?:\s+number)?|patient\s+id)[:\s#]*([A-Z0-9]{6,12})\b",
                score=0.85,
            ),
            Pattern(
                name="mrn_hash",
                regex=r"#([A-Z0-9]{7,10})\b",
                score=0.4,
            ),
        ],
        context=["mrn", "medical record", "chart", "patient id", "emr", "ehr"],
    )


def _icd_recognizer() -> PatternRecognizer:
    """ICD-10-CM codes: letter + 2 digits + optional dot + up to 4 more chars."""
    return PatternRecognizer(
        supported_entity="ICD_CODE",
        patterns=[
            Pattern(
                name="icd10",
                regex=r"\b[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?\b",
                score=0.55,
            ),
        ],
        context=["icd", "diagnosis", "dx", "code", "condition", "disease"],
    )


def _ndc_recognizer() -> PatternRecognizer:
    """National Drug Code — 5-4-2 or 5-4 numeric segments."""
    return PatternRecognizer(
        supported_entity="NDC_CODE",
        patterns=[
            Pattern(
                name="ndc",
                regex=r"\b\d{5}-\d{4}-\d{2}\b|\b\d{5}-\d{4}\b",
                score=0.75,
            ),
        ],
        context=["ndc", "drug code", "medication", "rx", "national drug"],
    )


def _health_plan_recognizer() -> PatternRecognizer:
    """Health plan / payer / insurance member IDs."""
    return PatternRecognizer(
        supported_entity="HEALTH_PLAN_ID",
        patterns=[
            Pattern(
                name="health_plan_id",
                regex=r"\b(?:member\s*id|plan\s*id|payer\s*id|policy\s*(?:number|no\.?))[:\s]*([A-Z0-9]{6,20})\b",
                score=0.75,
            ),
        ],
        context=["member", "plan", "payer", "insurance", "policy", "subscriber"],
    )


# ---------------------------------------------------------------------------
# Entity label normalisation
# ---------------------------------------------------------------------------

# Map Presidio entity types to shorter token prefixes
_ENTITY_PREFIX: dict[str, str] = {
    "PERSON":           "PERSON",
    "PHONE_NUMBER":     "PHONE",
    "EMAIL_ADDRESS":    "EMAIL",
    "LOCATION":         "LOCATION",
    "DATE_TIME":        "DATE",
    "US_SSN":           "SSN",
    "US_PASSPORT":      "PASSPORT",
    "US_DRIVER_LICENSE":"DL",
    "IBAN_CODE":        "IBAN",
    "CREDIT_CARD":      "CREDIT_CARD",
    "IP_ADDRESS":       "IP",
    "URL":              "URL",
    "MEDICAL_LICENSE":  "MED_LIC",
    "NPI_NUMBER":       "NPI",
    "MRN":              "MRN",
    "ICD_CODE":         "ICD",
    "NDC_CODE":         "NDC",
    "HEALTH_PLAN_ID":   "PLAN_ID",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DeIdResult:
    """
    Output of DeIdentifier.de_identify().

    Attributes
    ----------
    anonymized_text : str
        Input text with all detected PHI replaced by tokens.
    entity_map : dict[str, str]
        Mapping from each token (e.g. ``<PERSON_1>``) to the original value.
    entity_count : int
        Total number of entities detected.
    entities_by_type : dict[str, int]
        Count of detections per entity type.
    """
    anonymized_text: str
    entity_map: dict[str, str] = field(default_factory=dict)
    entity_count: int = 0
    entities_by_type: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DeIdentifier
# ---------------------------------------------------------------------------

# Entity types sent to the Presidio analyser
_ENTITIES: list[str] = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "LOCATION",
    "DATE_TIME",
    "US_SSN",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "IBAN_CODE",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "URL",
    "MEDICAL_LICENSE",
    "NPI_NUMBER",
    "MRN",
    "ICD_CODE",
    "NDC_CODE",
    "HEALTH_PLAN_ID",
]


class DeIdentifier:
    """
    PHI de-identifier backed by Presidio with healthcare-specific recognisers.

    Thread-safe: the Presidio AnalyzerEngine is stateless per call, so a single
    DeIdentifier instance can safely serve concurrent FastAPI request handlers.

    Args:
        score_threshold : Minimum Presidio confidence score to treat a span as
                          a PHI entity (default 0.35).  Lower values increase
                          recall at the cost of more false positives.
        language        : Language code passed to the analyser (default "en").
    """

    def __init__(
        self,
        score_threshold: float = 0.35,
        language: str = "en",
    ) -> None:
        self._threshold = score_threshold
        self._language  = language
        self._analyzer  = self._build_analyzer()
        logger.info("DeIdentifier initialised (threshold=%.2f)", score_threshold)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def de_identify(self, text: str) -> DeIdResult:
        """
        Detect and replace PHI entities in *text*.

        Returns a :class:`DeIdResult` with the anonymised text and a token→value
        entity_map.  If no entities are detected the input text is returned
        unchanged and entity_map is empty.

        Never raises — on any internal Presidio error the original text is
        returned with an empty entity_map so callers are not disrupted.
        """
        if not text or not text.strip():
            return DeIdResult(anonymized_text=text)

        try:
            results: list[RecognizerResult] = self._analyzer.analyze(
                text=text,
                entities=_ENTITIES,
                language=self._language,
                score_threshold=self._threshold,
            )
        except Exception as exc:
            logger.error("DeIdentifier: analyzer error — %s", exc)
            return DeIdResult(anonymized_text=text)

        if not results:
            return DeIdResult(anonymized_text=text)

        return self._replace_entities(text, results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_analyzer(self) -> AnalyzerEngine:
        """Construct AnalyzerEngine with spaCy NLP + custom healthcare recognisers."""
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        })
        nlp_engine = provider.create_engine()

        engine = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])

        for recognizer in [
            _medical_license_recognizer(),
            _npi_recognizer(),
            _mrn_recognizer(),
            _icd_recognizer(),
            _ndc_recognizer(),
            _health_plan_recognizer(),
        ]:
            engine.registry.add_recognizer(recognizer)

        return engine

    def _replace_entities(
        self, text: str, results: list[RecognizerResult]
    ) -> DeIdResult:
        """
        Replace detected spans with deterministic tokens, building entity_map.

        Overlapping spans are resolved by taking the highest-scoring span.
        Spans are processed right-to-left so character offsets stay valid after
        each substitution.
        """
        # Resolve overlaps: sort by score desc, greedily keep non-overlapping
        results_sorted = sorted(results, key=lambda r: r.score, reverse=True)
        kept: list[RecognizerResult] = []
        for r in results_sorted:
            if not any(r.start < k.end and r.end > k.start for k in kept):
                kept.append(r)

        # Sort kept spans right-to-left for in-place substitution
        kept.sort(key=lambda r: r.start, reverse=True)

        entity_map: dict[str, str] = {}
        counters: dict[str, int] = {}
        entities_by_type: dict[str, int] = {}

        chars = list(text)

        for r in kept:
            etype   = r.entity_type
            prefix  = _ENTITY_PREFIX.get(etype, etype)
            counters[prefix] = counters.get(prefix, 0) + 1
            token   = f"<{prefix}_{counters[prefix]}>"
            original = text[r.start:r.end]

            entity_map[token] = original
            entities_by_type[etype] = entities_by_type.get(etype, 0) + 1

            chars[r.start:r.end] = list(token)

        anonymized_text = "".join(chars)

        return DeIdResult(
            anonymized_text=anonymized_text,
            entity_map=entity_map,
            entity_count=len(kept),
            entities_by_type=entities_by_type,
        )
