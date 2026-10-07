"""Tissue relaxation and proton-density parameters at 3 T, with no geometry.

Units: ms. Proton density is relative, dermis = 1 by definition.
Every parameter is an interval with a confidence rating and a source; the
nominal value is a central point for sweeps, never the answer.
Citations were entered from memory and must be checked against the papers.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum


class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very low"


@dataclass(frozen=True)
class Param:
    """A tissue parameter as an interval with a nominal value, a confidence rating and a source."""

    lo: float
    hi: float
    nominal: float
    confidence: Confidence
    source: str

    def __post_init__(self) -> None:
        if not self.lo <= self.nominal <= self.hi:
            raise ValueError(f"need lo <= nominal <= hi, got {self.lo}, {self.nominal}, {self.hi}")


@dataclass(frozen=True)
class Tissue:
    """MR-visible pool of a tissue; an MT bound pool is to be added as a further optional field."""

    name: str
    t1_ms: Param
    t2_ms: Param
    t2star_ms: Param
    pd_rel: Param


H, M, L, VL = Confidence.HIGH, Confidence.MEDIUM, Confidence.LOW, Confidence.VERY_LOW

ARTERIAL_BLOOD = Tissue(
    name="arterial blood",
    t1_ms=Param(1600, 1950, 1650, M, "Lu et al., MRM 2004 (~1630 ms, Hct 0.42); "
                "Stanisz et al., MRM 2005 (~1930 ms); haematocrit-dependent"),
    t2_ms=Param(120, 275, 180, L, "oxygenation- and refocusing-interval-dependent; "
                "TRUST-type calibrations at Y~0.98; Stanisz et al., MRM 2005 (~275 ms)"),
    t2star_ms=Param(45, 75, 65, M, "Zhao et al., MRM 2007;58:592-597, Table 1: bovine blood "
                    "in a shimmed phantom; in vivo likely lower"),
    pd_rel=Param(1.1, 1.4, 1.25, L, "estimate: blood water ~0.8 g/mL vs dermis 0.6-0.7"),
)

VENOUS_BLOOD = Tissue(
    name="venous blood",
    t1_ms=Param(1550, 1900, 1600, M, "as arterial blood; deoxygenation shortens T1 slightly"),
    t2_ms=Param(40, 130, 80, L, "strongly oxygenation-dependent; superficial venous saturation "
                "0.6-0.85 assumed; refocusing-interval-dependent"),
    t2star_ms=Param(10, 50, 25, VL, "placeholder: intravascular T2* not reviewed"),
    pd_rel=Param(1.1, 1.4, 1.25, L, "estimate: as arterial blood"),
)

EPIDERMIS = Tissue(
    name="epidermis",
    t1_ms=Param(800, 1500, 1100, VL, "placeholder: no in vivo 3 T epidermis T1 reviewed"),
    t2_ms=Param(10, 40, 25, VL, "placeholder: no in vivo 3 T epidermis T2 reviewed"),
    t2star_ms=Param(5, 30, 15, VL, "placeholder: no in vivo 3 T epidermis T2* reviewed"),
    pd_rel=Param(0.6, 1.2, 0.9, VL, "placeholder: viable epidermis assumed dermis-like; "
                 "stratum corneum is near-invisible"),
)

DERMIS = Tissue(
    name="dermis",
    t1_ms=Param(900, 1500, 1200, VL, "placeholder: 3 T dermis literature not yet reviewed; "
                "nominal is the earlier codebase's implied value"),
    t2_ms=Param(20, 60, 45, VL, "placeholder: nominal is the earlier codebase's implied value; "
                "dermal T2 is multi-exponential (collagen-associated water)"),
    t2star_ms=Param(5, 30, 15, VL, "placeholder: prior results name this the dominant "
                    "uncertainty in the blood/dermis ratio"),
    pd_rel=Param(1.0, 1.0, 1.0, H, "reference tissue: dermis = 1 by definition"),
)

HYPODERMIS = Tissue(
    name="hypodermis (subcutaneous fat)",
    t1_ms=Param(360, 400, 380, M, "Gold et al., AJR 2004, subcutaneous fat at 3 T"),
    t2_ms=Param(60, 140, 70, L, "Gold et al., AJR 2004 (~70 ms); earlier codebase implied "
                "~130 ms; multi-peak fat T2 is method-dependent"),
    t2star_ms=Param(10, 50, 25, VL, "placeholder: mono-exponential T2* is ill-defined for "
                    "multi-peak fat at 3 T"),
    pd_rel=Param(1.1, 1.6, 1.4, L, "estimate: lipid ~100 mol 1H/L vs dermal water fraction "
                 "0.6-0.7 of 111 mol 1H/L; MR-visible dermal fraction unknown"),
)

MUSCLE = Tissue(
    name="muscle",
    t1_ms=Param(1400, 1450, 1420, M, "Stanisz et al., MRM 2005 (~1410 ms); "
                "Gold et al., AJR 2004 (~1420 ms)"),
    t2_ms=Param(30, 50, 40, M, "Gold et al., AJR 2004 (~32 ms); Stanisz et al., MRM 2005 (~50 ms)"),
    t2star_ms=Param(20, 35, 28, L, "approximate 3 T skeletal muscle range; not reviewed"),
    pd_rel=Param(1.0, 1.3, 1.15, L, "estimate: muscle water ~75 % vs dermis 60-70 %"),
)

ALL_TISSUES: tuple[Tissue, ...] = (
    ARTERIAL_BLOOD, VENOUS_BLOOD, EPIDERMIS, DERMIS, HYPODERMIS, MUSCLE,
)


def low_confidence_entries() -> list[tuple[str, str, Param]]:
    """Return (tissue name, parameter name, Param) for every LOW or VERY_LOW parameter."""
    weak = (Confidence.LOW, Confidence.VERY_LOW)
    return [
        (tissue.name, field.name, getattr(tissue, field.name))
        for tissue in ALL_TISSUES
        for field in fields(tissue)
        if isinstance(getattr(tissue, field.name), Param)
        and getattr(tissue, field.name).confidence in weak
    ]


def low_confidence_report() -> str:
    """Return one line per LOW or VERY_LOW parameter giving its range, nominal value and source."""
    return "\n".join(
        f"{tissue} {name}: {p.lo:g}-{p.hi:g} (nominal {p.nominal:g}) "
        f"{p.confidence.name}: {p.source}"
        for tissue, name, p in low_confidence_entries()
    )
