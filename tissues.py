"""Tissue, haemodynamic and acquisition parameters at 3 T, and the sampler that propagates their uncertainty.

Units: ms for T1, T2, T2*; mm/s for velocities; mm for diameters. Proton
density is relative, dermis = 1 by definition.

Every parameter is an interval with a confidence rating and a source. The
nominal value is a central point for sanity checks, never a published figure:
figures draw Latin-hypercube samples with a recorded seed (``sample``,
``sample_tissues``).

Sources are marked [checked] when the value has been read off the publisher's
record of that paper, and [unchecked] when it is reported here but the paper
itself has not been consulted. An [unchecked] source is a lead, not evidence.

Confidence refers to the INTERVAL, not the nominal: HIGH means a definition or
a tight direct measurement, MEDIUM a direct measurement of the right tissue at
3 T, LOW an indirect or single-source value, VERY_LOW a value with no
measurement of this tissue at this field behind it.

No measured quantity from this project's own images enters this module.
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

# Sources cited more than once.
STANISZ = ("Stanisz et al., Magn Reson Med 2005;54:507-512, doi:10.1002/mrm.20605, "
           "Table 1, 3 T, 37 C [checked]")
BOJORQUEZ = ("Bojorquez et al., Magn Reson Imaging 2017;35:69-80, doi:10.1016/j.mri.2016.08.021, "
             "a review pooling 3 T measurements across methods [checked]")
QUERLEUX = ("Querleux et al., in Imaging Technologies and Transdermal Delivery in Skin Disorders, "
            "Wiley-VCH 2020, ch. 14 [checked]")

ARTERIAL_BLOOD = Tissue(
    name="arterial blood",
    t1_ms=Param(1600, 1950, 1800, M,
                f"{STANISZ} gives 1932 +/- 85 ms. Lu et al., Magn Reson Med 2004 reports ~1630 ms at "
                "haematocrit 0.42 [unchecked]. T1 rises with haematocrit, and the interval spans both "
                "reports"),
    t2_ms=Param(120, 275, 180, L,
                f"{STANISZ} gives 275 +/- 50 ms. T2 falls steeply with deoxygenation and with the "
                "refocusing interval, so an in vivo arterial value at Y ~ 0.98 sits below the Stanisz "
                "figure; the lower bound allows for that"),
    t2star_ms=Param(45, 75, 65, M,
                    "Zhao et al., Magn Reson Med 2007;58:592-597, Table 1: bovine blood in a shimmed "
                    "phantom [unchecked]. In vivo values are lower, since the phantom removes "
                    "macroscopic field gradients"),
    pd_rel=Param(1.1, 1.4, 1.25, L,
                 "estimate, not a measurement: blood water content ~0.8 g/mL against a dermal water "
                 "fraction of 0.6-0.7"),
    fatsat_s=Param(1.0, 1.0, 1.0, H,
                   "a fat-selective pulse is off-resonance for blood water, which is on-resonance, so "
                   "it is left untouched"),
    mtc_s=Param(0.92, 1.0, 0.98, L,
                f"{STANISZ} measures a blood bound-pool fraction of 2.8 +/- 0.7 % and an MTR of "
                "11 +/- 4 %, so blood is weakly but not negligibly MT-active. Stanisz saturates "
                "continuously; a single MTC pulse per TR is far weaker, so the per-pulse residual sits "
                "near 1. The interval reflects that the mapping from MTR to a per-pulse residual is "
                "not established"),
)

VENOUS_BLOOD = Tissue(
    name="venous blood",
    t1_ms=Param(1550, 1900, 1750, M,
                f"as arterial blood ({STANISZ}); deoxygenation shortens T1 slightly"),
    t2_ms=Param(40, 130, 80, L,
                "strongly oxygenation-dependent, for a superficial venous saturation of 0.6-0.85, and "
                "dependent on the refocusing interval [unchecked]"),
    t2star_ms=Param(15, 32, 21, M,
                    "Zhao et al., Magn Reson Med 2007;58:592-597, Table 1: bovine blood in a shimmed "
                    "phantom [unchecked]. Deoxyhaemoglobin shortens T2* well below the arterial value"),
    pd_rel=Param(1.1, 1.4, 1.25, L, "estimate, as arterial blood"),
    fatsat_s=Param(1.0, 1.0, 1.0, H, "on-resonance water, as arterial blood"),
    mtc_s=Param(0.92, 1.0, 0.98, L, "as arterial blood"),
)

EPIDERMIS = Tissue(
    name="epidermis",
    t1_ms=Param(800, 1500, 1100, VL,
                "no peer-reviewed in vivo 3 T epidermis T1 located. Sharma, Nature Precedings 2010, "
                "doi:10101/npre.2010.4161.1 quotes T1 ~1000 ms without stating a field [checked, but "
                "a preprint and field not stated]. The layer is ~100 um thick, so partial volume "
                "dominates its contribution at the voxel sizes modelled here"),
    t2_ms=Param(10, 40, 25, VL,
                "no peer-reviewed in vivo 3 T epidermis T2 located. Sharma (above) quotes 20-35 ms "
                "for epidermis and dermis together"),
    t2star_ms=Param(5, 10, 8, VL,
                    "no measurement located; assumed to sit below the epidermis T2 lower bound, as "
                    "the invariant T2* <= T2 requires"),
    pd_rel=Param(0.6, 1.2, 0.9, VL,
                 "estimate: viable epidermis assumed dermis-like in water content, while the stratum "
                 "corneum is near-invisible at these echo times"),
    fatsat_s=Param(0.7, 1.0, 0.9, VL,
                   "an MT proxy, not spectral suppression: the off-resonance fat pulse partially "
                   "saturates macromolecule-bound protons, which exchange with free water, so the "
                   "value sits below 1 by an amount no measurement here constrains"),
    mtc_s=Param(0.5, 1.0, 0.8, VL, "as dermis; no epidermis-specific measurement located"),
)

DERMIS = Tissue(
    name="dermis",
    t1_ms=Param(850, 1400, 1100, L,
                f"{QUERLEUX} reports in vivo dermis T1 of 700-1000 ms at 1.5 T and 1000-1700 ms at "
                "7 T, citing Richard et al., J Invest Dermatol 1991 and Barral et al., Magn Reson Med "
                "2010;63:790-796 [both unchecked]. No 3 T value was located, so the interval is the "
                "field interpolation between those two brackets. Barral measures skin at 3 T and "
                "should be read directly"),
    t2_ms=Param(20, 60, 35, L,
                f"{QUERLEUX} describes dermal T2 as tens of milliseconds. Dermal T2 is multi-"
                "exponential, with a dominant short component from collagen-bound water, so the "
                "number depends on how it is fitted and on the echo spacing"),
    t2star_ms=Param(5, 20, 12, L,
                    "Sharma, Nature Precedings 2010, doi:10101/npre.2010.4161.1 reports a single-"
                    "component dermal T2* of 12 +/- 1.4 ms over echo times 2.8-20 ms, and a "
                    "bi-exponential fit of 9.8 ms carrying ~91 % of the signal with a 42.7 ms "
                    "remainder [checked, but a preprint]. Sensitivity analyses consistently make this "
                    "the dominant parameter for blood/dermis contrast, so it warrants a direct "
                    "measurement"),
    pd_rel=Param(1.0, 1.0, 1.0, H, "reference tissue: dermis = 1 by definition"),
    fatsat_s=Param(0.7, 1.0, 0.9, VL,
                   "an MT proxy, not spectral suppression: the off-resonance fat pulse partially "
                   "saturates collagen-bound protons, which exchange with free water, so the value "
                   "sits below 1 by an amount no measurement here constrains"),
    mtc_s=Param(0.5, 1.0, 0.8, VL,
                f"the bound-pool fraction of real dermis is not established. {QUERLEUX} measures "
                "0.66-1.37 % in RECONSTRUCTED dermis at 4.7 T, but those samples are 90-96 % water "
                "and far more dilute than dermis in vivo, so the figure is a floor. For scale, "
                f"{STANISZ} gives 13.9 % for white matter and 17-18 % for cartilage. The interval is "
                "wide because collagen-rich tissue could plausibly sit anywhere between those"),
)

HYPODERMIS = Tissue(
    name="hypodermis (subcutaneous fat)",
    t1_ms=Param(300, 450, 380, M,
                f"{BOJORQUEZ} tabulates eleven 3 T measurements of subcutaneous fat spanning "
                "253-450 ms, including 371 +/- 8 ms (Gold et al., AJR 2004), 382 +/- 13 ms "
                "(De Bazelaire et al.) and 421 +/- 104 ms (Barral et al.). The interval covers the "
                "bulk of that spread"),
    t2_ms=Param(45, 160, 90, L,
                f"{BOJORQUEZ} tabulates 3 T fat T2 from 41 to 371 ms and states the value is strongly "
                "method-dependent: multi-echo spin echo gives ~113-154 ms, single spin echo ~41 ms, "
                "fast spin echo rising with echo-train length. The spread is dominated by method, not "
                "by biology"),
    t2star_ms=Param(10, 45, 25, VL,
                    "no 3 T subcutaneous-fat T2* located. A mono-exponential T2* is in any case "
                    "ill-defined for fat, whose several spectral peaks dephase against each other"),
    pd_rel=Param(1.1, 1.6, 1.4, L,
                 "estimate, not a measurement: lipid carries ~100 mol 1H/L against a dermal water "
                 "fraction of 0.6-0.7 of 111 mol 1H/L. The MR-visible dermal fraction is unknown"),
    fatsat_s=Param(0.05, 0.5, 0.15, L,
                   "achievable CHESS suppression rather than ideal suppression. The olefinic "
                   "(5.3 ppm) and glycerol (~4.2 ppm) peaks together carry ~8-10 % of fat protons "
                   "and lie too close to water to be suppressed by a fat-selective pulse (Ren et al., "
                   "J Lipid Res 2008; Hamilton et al., NMR Biomed 2011 [both unchecked]), which sets "
                   "the floor. B0 inhomogeneity over a limb shifts the main methylene peak partly out "
                   "of the pulse band (Bley et al., J Magn Reson Imaging 2010 [unchecked]), which "
                   "sets the upper bound"),
    mtc_s=Param(0.9, 1.0, 0.97, L,
                "lipid protons have little exchange with a macromolecular pool, so an off-resonance "
                "MT pulse saturates fat only weakly. No direct measurement of a per-pulse residual "
                "for subcutaneous fat was located"),
)

MUSCLE = Tissue(
    name="muscle",
    t1_ms=Param(1000, 1500, 1400, M,
                f"{STANISZ} gives 1412 +/- 13 ms and Gold et al., AJR 2004 gives 1420 +/- 38 ms, but "
                f"{BOJORQUEZ} tabulates 3 T skeletal muscle from 898 to 1509 ms across methods. The "
                "interval spans the bulk of that spread rather than the two tightest reports"),
    t2_ms=Param(27, 50, 35, M,
                f"{BOJORQUEZ} tabulates 27-44 ms across methods, including 32 +/- 2 ms (Gold et al., "
                f"AJR 2004); {STANISZ} gives 50 +/- 4 ms"),
    t2star_ms=Param(15, 27, 22, L,
                    "no single reviewed 3 T skeletal-muscle T2* source; bounded above by the muscle "
                    "T2 lower bound, as the invariant T2* <= T2 requires"),
    pd_rel=Param(1.0, 1.3, 1.15, L,
                 "estimate, not a measurement: muscle water content ~75 % against a dermal 60-70 %"),
    fatsat_s=Param(1.0, 1.0, 1.0, H,
                   "on-resonance water. Muscle is MT-active, so by the same argument made for dermis "
                   "it may sit slightly below 1"),
    mtc_s=Param(0.6, 0.95, 0.8, L,
                f"{STANISZ} measures a muscle bound-pool fraction of 7.4 +/- 1.3 %, well above blood "
                "and below cartilage, so muscle is appreciably MT-active. Its reported MTR of "
                "88 +/- 2 % is for continuous saturation and does not transfer to a per-pulse "
                "residual under a single MTC pulse per TR; the interval is correspondingly wide"),
)

TISSUES: dict[str, Tissue] = {
    "arterial_blood": ARTERIAL_BLOOD, "venous_blood": VENOUS_BLOOD, "epidermis": EPIDERMIS,
    "dermis": DERMIS, "hypodermis": HYPODERMIS, "muscle": MUSCLE,
}
ALL_TISSUES: tuple[Tissue, ...] = tuple(TISSUES.values())

# --- Haemodynamics -------------------------------------------------------------
# Direct human measurements exist for capillaries and for ~2 mm perforators.
# Everything between them is interpolated.

CAPILLARY_VELOCITY = Param(
    0.3, 1.0, 0.6, M,
    "nailfold capillaroscopy of 10-15 um vessels: ~0.84 mm/s in the arteriolar limb and ~0.47 mm/s "
    "in the venular limb [unchecked]. Methods disagree by up to an order of magnitude. Used to "
    "anchor the velocity-diameter scaling, not as a target-vessel value", log=True)

ARTERIOLE_VELOCITY = Param(
    1.0, 5.0, 2.5, L,
    "interpolated from the capillary and perforator anchors for ascending arterioles and the "
    "subpapillary plexus, ~20-50 um; no direct measurement located", log=True)

TARGET_VESSEL_VELOCITY = Param(
    5.0, 50.0, 15.0, L,
    "time-averaged mean velocity of the 100-500 um target vessels. No direct measurement in human "
    "subcutaneous vessels of this calibre was located: they fall between what capillaroscopy and "
    "what duplex ultrasound can reach. The nominal is therefore interpolated between the two "
    "calibres that are measured. Lower anchor: nailfold capillaroscopy, ~12.5 um at ~0.65 mm/s "
    "(CAPILLARY_VELOCITY). Upper anchor: lower-limb cutaneous perforators of mean diameter 1.93 mm "
    "at peak systolic 16.9 cm/s with resistance index 0.71, a time-averaged mean of ~89 mm/s "
    "(J Reconstr Microsurg 2021;37(1):75-82, doi:10.1055/s-0040-1702162 [checked]). A power law "
    "through those anchors, 154x apart in diameter, gives v ~ d^0.98, indistinguishable from the "
    "v ~ d of Murray's law, which is the scaling expected on theoretical grounds. That yields "
    "4.9 mm/s at 100 um, 14.5 at 300 um and 23.8 at 500 um. The interval is wider than that "
    "4.9-23.8 span, which is only the diameter sweep and carries no uncertainty in the exponent, "
    "in the peak-systolic-to-mean conversion, between arteries and veins, or between subjects. "
    "Sampled independently of TARGET_DIAMETER although v ~ d couples them", log=True)

THERMOREGULATORY_FACTOR = Param(
    0.3, 3.0, 1.0, M,
    "skin perfusion swings by about an order of magnitude with thermoregulation; a cool scanner "
    "room and an immobilised limb sit at the low end. Multiplies the vessel velocity", log=True)

TARGET_DIAMETER = Param(
    0.1, 1.0, 0.3, H,
    "mm. A swept axis rather than an uncertainty: spans the 100-500 um target and small cutaneous "
    "arteries and veins up to 1 mm", log=True)

VESSEL_ANGLE_TO_SLAB_NORMAL = Param(
    0.0, 90.0, 75.0, M,
    "deg. 0 = the vessel crosses the slab end-on, 90 = it lies in the slab plane and never leaves "
    "it. The dermal and subpapillary plexuses run near-parallel to the skin surface")

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
        -0.31, 0.31, 0.0, L,
        "fraction of Mz one travelling-saturation-band pulse leaves: cos(90 deg x b) for a B1 scale "
        "b of 0.8-1.2 under body transmit at the leg. Negative means over-tipped. Assumes the band "
        "is played once per TR"),
    "grappa_r2_g_factor": Param(
        1.0, 1.5, 1.15, L,
        "GRAPPA R = 2 noise amplification for a 4-channel surface array. Few channels and little "
        "coil-sensitivity variation along the phase direction push g above 1, and it varies with "
        "depth (Pruessmann et al., Magn Reson Med 1999; Breuer et al., Magn Reson Med 2009 [both "
        "unchecked]). Not derivable without coil sensitivity maps"),
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


def unchecked_sources() -> list[tuple[str, str]]:
    """Return (owner, parameter name) for every parameter whose source has not been read at the publisher."""
    entries = [(t.name, f, getattr(t, f)) for t in ALL_TISSUES for f in PARAM_FIELDS]
    entries += [("haemodynamics", k, p) for k, p in HAEMODYNAMICS.items()]
    entries += [("acquisition", k, p) for k, p in ACQUISITION.items()]
    return [(owner, name) for owner, name, p in entries if "[unchecked]" in p.source]
