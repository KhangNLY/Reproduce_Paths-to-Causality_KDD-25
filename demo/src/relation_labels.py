"""Readable English and Vietnamese names for Hetionet v1.0 metaedge abbreviations."""
from __future__ import annotations

# The abbreviations/definitions follow Hetionet's describe/edges/metaedges.tsv.
RELATION_LABELS: dict[str, tuple[str, str]] = {
    "AdG": ("downregulates", "điều hòa giảm"),
    "AeG": ("expresses", "biểu hiện"),
    "AuG": ("upregulates", "điều hòa tăng"),
    "CbG": ("binds", "liên kết với"),
    "CcSE": ("causes", "gây ra"),
    "CdG": ("downregulates", "điều hòa giảm"),
    "CpD": ("palliates", "làm giảm nhẹ"),
    "CrC": ("resembles", "tương tự"),
    "CtD": ("treats", "điều trị"),
    "CuG": ("upregulates", "điều hòa tăng"),
    "DaG": ("associates", "liên quan với"),
    "DdG": ("downregulates", "điều hòa giảm"),
    "DlA": ("localizes", "định vị tại"),
    "DpS": ("presents", "biểu hiện"),
    "DrD": ("resembles", "tương tự"),
    "DuG": ("upregulates", "điều hòa tăng"),
    "GcG": ("covaries", "đồng biến với"),
    "GiG": ("interacts", "tương tác với"),
    "GpBP": ("participates in", "tham gia vào"),
    "GpCC": ("participates in", "tham gia vào"),
    "GpMF": ("participates in", "tham gia vào"),
    "GpPW": ("participates in", "tham gia vào"),
    "Gr>G": ("regulates", "điều hòa"),
    "PCiC": ("includes", "bao gồm"),
}

def _label_key(code: str) -> str:
    key = str(code or "").split("_")[-1]
    return key.replace(">", "")


def _label(code: str) -> tuple[str, str]:
    return RELATION_LABELS.get(code, RELATION_LABELS.get(_label_key(code), (code or "related to", code or "liên quan đến")))


def relation_verb(code: str) -> str:
    """Return a compact English relation label for an edge abbreviation."""
    return _label(code)[0]


def relation_verb_vi(code: str) -> str:
    """Return a compact Vietnamese relation label for an edge abbreviation."""
    return _label(code)[1]


def relation_description(code: str, source_kind: str, target_kind: str) -> tuple[str, str]:
    """Return detailed EN/VI tooltips for an edge, with the real stored direction."""
    verb_en, verb_vi = _label(code)
    source_kind = source_kind or "Entity"
    target_kind = target_kind or "Entity"
    return (
        f"{source_kind} {verb_en} {target_kind} ({code})",
        f"{source_kind} {verb_vi} {target_kind} ({code})",
    )
