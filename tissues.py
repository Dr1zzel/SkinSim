"""Tissue, haemodynamic and acquisition parameters at 3 T, and the sampler that propagates their uncertainty.

Units: ms for T1, T2, T2*; mm/s for velocities; mm for diameters. Proton
density is relative, dermis = 1 by definition.
Every parameter is an interval with a confidence rating and a source; the
nominal value is a central point for sanity checks, never the answer, and
never a published figure: figures draw Latin-hypercube samples with a
recorded seed (``sample``, ``sample_tissues``).
Citations were entered from memory and must be checked against the papers.
Nothing here is set from the 8 or 13 October 2026 measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike


class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very low"


@dataclass(frozen=True)
class Param:
    """A parameter as an interval with a nominal value, a confidence rating, a source and a sampling scale."""

    lo: float
    hi: float
    nominal: float
    confidence: Confidence
    source: str
    log: bool = False  # sample log-uniformly, for quantities spanning decades

    def __post_init__(self) -> None:
        if not self.lo <= self.nominal <= self.hi:
            raise ValueError(f"need lo <= nominal <= hi, got {self.lo}, {self.nominal}, {self.hi}")
        if self.log and self.lo <= 0:
            raise ValueError("log sampling needs lo > 0")


@dataclass(frozen=True)
class Tissue:
    """MR-visible pool of a tissue; an MT bound pool is to be added as a further optional field."""

    name: str
    t1_ms: Param
    t2_ms: Param
    t2star_ms: Param  # capped at the T2 lower bound so no draw has T2* > T2
    pd_rel: Param
    fatsat_s: Param  # fraction of Mz one fat-saturation pulse leaves (sequences/prep.py)
    mtc_s: Param  # fraction of Mz one MTC pulse leaves: an effective, crude MT factor


PARAM_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Tissue) if f.name != "name")


@dataclass(frozen=True)
class TissueSample:
    """Values a simulation runs on: one point from a tissue's intervals, or arrays of draws along a sample axis."""

    name: str
    t1_ms: ArrayLike
    t2_ms: ArrayLike
    t2star_ms: ArrayLike
    pd_rel: ArrayLike
    fatsat_s: ArrayLike = 1.0  # 1 = untouched; nominal() and the sampler always fill these from Tissue
    mtc_s: ArrayLike = 1.0

    @classmethod
    def nominal(cls, tissue: Tissue) -> TissueSample:
        """Return the tissue's nominal values, for sanity checks only."""
        return cls(tissue.name, **{f: getattr(tissue, f).nominal for f in PARAM_FIELDS})

    def at(self, i: int) -> TissueSample:
        """Return draw i as a scalar sample, for code that caches on hashable samples."""
        return TissueSample(self.name, **{f: float(np.asarray(getattr(self, f))[i]) for f in PARAM_FIELDS})


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
    fatsat_s=Param(1.0, 1.0, 1.0, H, "on-resonance water: a fat-selective pulse leaves blood "
                   "untouched, and blood's bound pool is negligible"),
    mtc_s=Param(0.95, 1.0, 0.98, M, "blood bound-pool fraction < 1 %, so an off-resonance MT pulse "
                "barely saturates it (SkinMRA; Stanisz et al., MRM 2005, for MT-inactive fluids)"),
)

VENOUS_BLOOD = Tissue(
    name="venous blood",
    t1_ms=Param(1550, 1900, 1600, M, "as arterial blood; deoxygenation shortens T1 slightly"),
    t2_ms=Param(40, 130, 80, L, "strongly oxygenation-dependent; superficial venous saturation "
                "0.6-0.85 assumed; refocusing-interval-dependent"),
    t2star_ms=Param(15, 32, 21, M, "Zhao et al., MRM 2007;58:592-597, Table 1: bovine blood "
                    "in a shimmed phantom; in vivo likely lower"),
    pd_rel=Param(1.1, 1.4, 1.25, L, "estimate: as arterial blood"),
    fatsat_s=Param(1.0, 1.0, 1.0, H, "on-resonance water: a fat-selective pulse leaves blood "
                   "untouched, and blood's bound pool is negligible"),
    mtc_s=Param(0.95, 1.0, 0.98, M, "blood bound-pool fraction < 1 %, so an off-resonance MT pulse "
                "barely saturates it (SkinMRA; Stanisz et al., MRM 2005, for MT-inactive fluids)"),
)

EPIDERMIS = Tissue(
    name="epidermis",
    t1_ms=Param(800, 1500, 1100, VL, "placeholder: no in vivo 3 T epidermis T1 reviewed"),
    t2_ms=Param(10, 40, 25, VL, "placeholder: no in vivo 3 T epidermis T2 reviewed"),
    t2star_ms=Param(5, 10, 8, VL, "placeholder: no in vivo 3 T epidermis T2* reviewed; "
                    "capped at the T2 lower bound"),
    pd_rel=Param(0.6, 1.2, 0.9, VL, "placeholder: viable epidermis assumed dermis-like; "
                 "stratum corneum is near-invisible"),
    fatsat_s=Param(0.7, 1.0, 0.9, VL, "MT PROXY, NOT SPECTRAL SUPPRESSION: the off-resonance fat "
                   "pulse partially saturates collagen-bound protons, which exchange with free water, so "
                   "epidermis sits below 1 by an unknown amount. One of the quantities the measurements will "
                   "constrain, not one they have constrained"),
    mtc_s=Param(0.5, 0.95, 0.75, VL, "effective per-pulse residual under the MTC pulse; skin is "
                "strongly MT-active (Mirrashed & Sharp, Skin Res Technol 2004, qualitative only); no "
                "quantitative 3 T value for this pulse located"),
)

DERMIS = Tissue(
    name="dermis",
    t1_ms=Param(900, 1500, 1200, VL, "placeholder: 3 T dermis literature not yet reviewed; "
                "nominal is the earlier codebase's implied value"),
    t2_ms=Param(20, 60, 45, VL, "placeholder: nominal is the earlier codebase's implied value; "
                "dermal T2 is multi-exponential (collagen-associated water)"),
    t2star_ms=Param(5, 20, 15, VL, "placeholder: prior results name this the dominant "
                    "uncertainty in the blood/dermis ratio; capped at the T2 lower bound"),
    pd_rel=Param(1.0, 1.0, 1.0, H, "reference tissue: dermis = 1 by definition"),
    fatsat_s=Param(0.7, 1.0, 0.9, VL, "MT PROXY, NOT SPECTRAL SUPPRESSION: the off-resonance fat "
                   "pulse partially saturates collagen-bound protons, which exchange with free water, so "
                   "dermis sits below 1 by an unknown amount. One of the quantities the measurements will "
                   "constrain, not one they have constrained"),
    mtc_s=Param(0.5, 0.95, 0.75, VL, "effective per-pulse residual under the MTC pulse; skin is "
                "strongly MT-active (Mirrashed & Sharp, Skin Res Technol 2004, qualitative only); no "
                "quantitative 3 T value for this pulse located"),
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
    fatsat_s=Param(0.05, 0.5, 0.15, L, "achievable CHESS suppression: the olefinic (5.3 ppm) and "
                   "glycerol (~4.2 ppm) peaks, together ~8-10 % of fat protons, lie close to water and "
                   "escape a fat-selective pulse (Ren et al., J Lipid Res 2008; Hamilton et al., NMR "
                   "Biomed 2011), which sets the floor; B0 inhomogeneity over a limb shifts the main "
                   "methylene peak partly out of the pulse band (Bley et al., JMRI 2010), which widens "
                   "the upper bound. Set from physics, not from the images"),
    mtc_s=Param(0.9, 1.0, 0.97, L, "fat has a small bound pool, so little MT saturation"),
)

MUSCLE = Tissue(
    name="muscle",
    t1_ms=Param(1400, 1450, 1420, M, "Stanisz et al., MRM 2005 (~1410 ms); "
                "Gold et al., AJR 2004 (~1420 ms)"),
    t2_ms=Param(30, 50, 40, M, "Gold et al., AJR 2004 (~32 ms); Stanisz et al., MRM 2005 (~50 ms)"),
    t2star_ms=Param(20, 30, 26, L, "approximate 3 T skeletal muscle range, not reviewed; "
                    "capped at the T2 lower bound"),
    pd_rel=Param(1.0, 1.3, 1.15, L, "estimate: muscle water ~75 % vs dermis 60-70 %"),
    fatsat_s=Param(1.0, 1.0, 1.0, H, "on-resonance water, as specified. By the dermis MT argument "
                   "muscle, also MT-active, may sit slightly below 1"),
    mtc_s=Param(0.6, 0.95, 0.8, L, "muscle MTR under typical off-resonance MT pulses is ~20-40 % at "
                "steady state; the per-pulse effective residual is inferred from that, crudely"),
)

TISSUES: dict[str, Tissue] = {
    "arterial_blood": ARTERIAL_BLOOD, "venous_blood": VENOUS_BLOOD, "epidermis": EPIDERMIS,
    "dermis": DERMIS, "hypodermis": HYPODERMIS, "muscle": MUSCLE,
}
ALL_TISSUES: tuple[Tissue, ...] = tuple(TISSUES.values())

# --- Haemodynamics -------------------------------------------------------------
# Ported from SkinMRA (skinmra_sim/tissues.py). Direct human measurements exist
# for capillaries; everything above them is inferred by scaling.

CAPILLARY_VELOCITY = Param(
    0.3, 1.0, 0.6, M, "nailfold capillaroscopy: arteriolar limb ~0.84 mm/s, venular limb "
    "~0.47 mm/s; methods disagree ~10x. 10-15 um vessels: anchors the scaling only", log=True)

ARTERIOLE_VELOCITY = Param(
    1.0, 5.0, 2.5, L, "inferred; ascending arterioles and plexus, ~20-50 um", log=True)

TARGET_VESSEL_VELOCITY = Param(
    5.0, 25.0, 14.0, L,
    "mm/s. Time-averaged mean velocity of the 100-500 um target vessels (SKIN-COILS WP3). "
    "Revised DOWN 16 Oct 2026 from 10-150 mm/s, and not from any image: that entry was "
    "explicitly written for 0.3-1 mm vessels and was then being applied to a target one "
    "calibre band smaller. No direct measurement of velocity in 100-500 um human subcutaneous "
    "vessels was found - they fall between what capillaroscopy and what duplex can reach - so "
    "the range is BRACKETED between the two calibres that are measured, and interpolated. "
    "Lower anchor: nailfold capillaroscopy, ~12.5 um at ~0.65 mm/s (CAPILLARY_VELOCITY above). "
    "Upper anchor: lower-limb cutaneous perforators, mean diameter 1.93 mm, peak systolic "
    "16.9 cm/s with resistance index 0.71, so a time-averaged mean of ~89 mm/s (Visconti et al., "
    "J Reconstr Microsurg 2021;37(1):75-82, doi:10.1055/s-0040-1702162). A power law through "
    "those two anchors, 154x apart in diameter, gives v ~ d^0.98 - indistinguishable from the "
    "v ~ d of Murray's law, which is the scaling expected on theoretical grounds and is the "
    "reason to trust the interpolation. That yields 4.9 mm/s at 100 um and 23.8 mm/s at 500 um. "
    "Still LOW confidence: interpolated, arterial, and resting. Swept, never a fixed assumption",
    log=True)

THERMOREGULATORY_FACTOR = Param(
    0.3, 3.0, 1.0, M, "skin perfusion swings an order of magnitude with thermoregulation; "
    "a cool scanner room and an immobilised leg sit at the low end. Multiplies velocity", log=True)

TARGET_DIAMETER = Param(
    0.1, 1.0, 0.3, H, "mm. Swept axis, not an uncertainty: spans the SKIN-COILS proposal's "
    "100-500 um target and small cutaneous arteries and veins up to 1 mm", log=True)

VESSEL_ANGLE_TO_SLAB_NORMAL = Param(
    0.0, 90.0, 75.0, M, "deg. 0 = through-slab (brain-like), 90 = in the slab plane. The dermal "
    "and subpapillary plexuses run near-parallel to the skin surface")

HAEMODYNAMICS: dict[str, Param] = {
    "capillary_velocity": CAPILLARY_VELOCITY,
    "arteriole_velocity": ARTERIOLE_VELOCITY,
    "target_vessel_velocity": TARGET_VESSEL_VELOCITY,
    "thermoregulatory_factor": THERMOREGULATORY_FACTOR,
    "target_diameter": TARGET_DIAMETER,
    "vessel_angle_to_slab_normal": VESSEL_ANGLE_TO_SLAB_NORMAL,
}

# Uncertain quantities set by the hardware rather than the tissue, ranged and
# sampled like everything else.
ACQUISITION: dict[str, Param] = {
    "sat_band_pulse_residual": Param(
        -0.31, 0.31, 0.0, L, "fraction of Mz one travelling-saturation-band pulse leaves, cos(90 deg x b) "
        "for a B1 scale b of 0.8-1.2 under body transmit at the leg; negative = over-tipped. Assumes the "
        "band is played every TR, which is not verified from the protocol"),
    "grappa_r2_g_factor": Param(
        1.0, 1.5, 1.15, L, "GRAPPA R = 2 noise amplification for a 4-channel surface array: few "
        "channels and little coil-sensitivity variation along the phase direction push g above 1, "
        "and it varies with depth (Pruessmann et al., MRM 1999; Breuer et al., MRM 2009). Not "
        "derivable without coil sensitivity maps; wide interval"),
}


def signal_weighting(tissue: TissueSample, te_ms: ArrayLike) -> ArrayLike:
    """Return the gradient-echo proton-density and T2* weighting pd * exp(-TE / T2*), exact for engines run with unit M0."""
    te = np.asarray(te_ms, dtype=float)
    return np.asarray(tissue.pd_rel, dtype=float) * np.exp(-te / np.asarray(tissue.t2star_ms, dtype=float))


def collect_params(
    tissue_keys: list[str], tissue_fields: tuple[str, ...] = PARAM_FIELDS,
    haemodynamic_keys: tuple[str, ...] = (), acquisition_keys: tuple[str, ...] = (),
) -> dict[str, Param]:
    """Return selected parameters flattened into one ordered dict keyed '<tissue>.<field>' or by bare key."""
    out = {f"{k}.{f}": getattr(TISSUES[k], f) for k in tissue_keys for f in tissue_fields}
    out.update({k: HAEMODYNAMICS[k] for k in haemodynamic_keys})
    out.update({k: ACQUISITION[k] for k in acquisition_keys})
    return out


def sample(params: dict[str, Param], n: int, seed: int,
           method: Literal["lhs", "sobol", "nominal"] = "lhs") -> dict[str, np.ndarray]:
    """Return n deterministic draws per parameter over its interval (Sobol rounds n up to a power of two)."""
    from scipy.stats import qmc  # imported here: scipy.stats is slow to import and only sampling needs it

    if method == "nominal":
        return {k: np.array([p.nominal]) for k, p in params.items()}
    if method == "lhs":
        unit = qmc.LatinHypercube(d=len(params), seed=seed).random(n)
    elif method == "sobol":
        unit = qmc.Sobol(d=len(params), scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(max(n, 2)))))
    else:
        raise ValueError(f"unknown method {method!r}")
    return {k: _scale(p, unit[:, j]) for j, (k, p) in enumerate(params.items())}


def _scale(p: Param, u: np.ndarray) -> np.ndarray:
    if p.log:
        return np.exp(np.log(p.lo) + u * (np.log(p.hi) - np.log(p.lo)))
    return p.lo + u * (p.hi - p.lo)


def sample_tissues(
    tissue_keys: list[str], n: int, seed: int, method: Literal["lhs", "sobol", "nominal"] = "lhs",
    haemodynamic_keys: tuple[str, ...] = (), acquisition_keys: tuple[str, ...] = (),
) -> tuple[dict[str, TissueSample], dict[str, np.ndarray]]:
    """Return one jointly drawn TissueSample per tissue key, with arrays along the sample axis, and the other draws."""
    draws = sample(collect_params(tissue_keys, PARAM_FIELDS, haemodynamic_keys, acquisition_keys),
                   n, seed, method)
    tissues = {k: TissueSample(TISSUES[k].name, **{f: draws[f"{k}.{f}"] for f in PARAM_FIELDS})
               for k in tissue_keys}
    others = {k: v for k, v in draws.items() if "." not in k}
    return tissues, others


def low_confidence_entries() -> list[tuple[str, str, Param]]:
    """Return (owner, parameter name, Param) for every LOW or VERY_LOW parameter, tissues first."""
    weak = (Confidence.LOW, Confidence.VERY_LOW)
    entries = [(t.name, f, getattr(t, f)) for t in ALL_TISSUES for f in PARAM_FIELDS]
    entries += [("haemodynamics", k, p) for k, p in HAEMODYNAMICS.items()]
    entries += [("acquisition", k, p) for k, p in ACQUISITION.items()]
    return [e for e in entries if e[2].confidence in weak]


def low_confidence_report() -> str:
    """Return one line per LOW or VERY_LOW parameter giving its range, nominal value and source."""
    return "\n".join(
        f"{owner} {name}: {p.lo:g}-{p.hi:g} (nominal {p.nominal:g}) {p.confidence.name}: {p.source}"
        for owner, name, p in low_confidence_entries()
    )
