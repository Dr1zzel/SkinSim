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
    fatsat_s: Param  # fraction of Mz one fat-saturation pulse leaves (sequence.SaturationModule)
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
RICHARD = ("Richard et al., J Invest Dermatol 1991;97(1):120-125, Table I: 1.5 T, 9 healthy subjects, 5 ROI each [checked]")
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
    t1_ms=Param(850, 1300, 1050, L,
                f"{RICHARD} measures 887 +/- 92 ms in calf and 720 +/- 104 ms in heel at 1.5 T. No "
                "3 T value located; T1 lengthens with field, so the interval is shifted up from the "
                "1.5 T measurement rather than taken from it. The layer is ~100 um thick, so partial "
                "volume dominates its contribution at the voxel sizes modelled here"),
    t2_ms=Param(15, 40, 25, L,
                f"{RICHARD} measures 22.3 +/- 7 ms in calf and 35.6 +/- 5 ms in heel at 1.5 T; the "
                "site difference is significant (p < 0.01) and the lower leg is the target. T2 is "
                "only weakly field-dependent, so the 1.5 T value transfers. Note epidermal T2 is "
                "LONGER than dermal, which the same paper attributes to the absence of collagen"),
    t2star_ms=Param(5, 15, 10, VL,
                    "no measurement located; bounded above by the epidermis T2 lower bound, as the "
                    "invariant T2* <= T2 requires"),
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
                f"{RICHARD} measures 870 +/- 143 ms in calf and 728 +/- 104 ms in heel at 1.5 T, "
                f"consistent with the 700-1000 ms bracket {QUERLEUX} quotes for 1.5 T against "
                "1000-1700 ms at 7 T. No 3 T measurement was located, so the interval is the field "
                "interpolation between those brackets. Barral et al., Magn Reson Med 2010;63:790-796 "
                "images skin at 3 T but reports no relaxation times [checked]; the measurement is in "
                "its reference 39, Barral et al., Skin T1 mapping at 1.5T, 3T and 7T, Proc ISMRM 2009 "
                "p. 4451, which would settle this"),
    t2_ms=Param(10, 23, 14, M,
                f"{RICHARD} measures 13 +/- 2.4 ms in calf and 17.9 +/- 5 ms in heel at 1.5 T. The "
                "site difference is significant (p < 0.01) and the lower leg is the target, so the "
                "interval is weighted to the calf value. T2 is only weakly field-dependent. Dermal T2 "
                "is short because collagen is ~70 % of dermal dry weight and constrains water proton "
                "mobility; it is also multi-exponential, so the fitted value depends on echo spacing"),
    t2star_ms=Param(4, 10, 8, L,
                    "Sharma, Nature Precedings 2010, doi:10101/npre.2010.4161.1 reports a single-"
                    "component dermal T2* of 12 +/- 1.4 ms over echo times 2.8-20 ms, and a "
                    "bi-exponential fit of 9.8 ms carrying ~91 % of the signal with a 42.7 ms "
                    "remainder [checked, but a preprint]. That 12 ms sits above the dermis T2 lower "
                    "bound measured by Richard, which T2* cannot exceed, so the interval is capped "
                    "there and the bi-exponential fast component is the better guide. Sensitivity "
                    "analyses consistently make this the dominant parameter for blood/dermis "
                    "contrast, so it warrants a direct measurement"),
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
    t2_ms=Param(30, 160, 70, L,
                f"{RICHARD} measures 35.4 +/- 3.6 ms in calf at 1.5 T by spin echo. {BOJORQUEZ} "
                "tabulates 3 T fat T2 from 41 to 371 ms and states the value is strongly "
                "method-dependent: multi-echo spin echo gives ~113-154 ms, single spin echo ~41 ms, "
                "fast spin echo rising with echo-train length. The spread is dominated by method, not "
                "by biology"),
    t2star_ms=Param(10, 30, 20, VL,
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
    "in the peak-systolic-to-mean conversion or between subjects. Both anchors are ARTERIAL, so "
    "this interval applies to the arterial half of the target population only; see "
    "VENOUS_VESSEL_VELOCITY. Sampled independently of TARGET_DIAMETER although v ~ d couples them",
    log=True)

VENOUS_VESSEL_VELOCITY = Param(
    1.4, 7.8, 3.5, M,
    "time-averaged mean velocity of 100-500 um SUBCUTANEOUS VEINS, supine. The arterial scaling "
    "above does not transfer: applied to a 3 mm vessel it predicts 134 mm/s, whereas the great "
    "saphenous vein of legs without varicose veins runs at 18.0 mm/s (IQR 7.6-30.3), measured "
    "supine by phase-contrast MRI in 53 legs (Tseng et al., Diagnostics 2022;12:118, "
    "doi:10.3390/diagnostics12010118, Table 3 [checked]) - an over-prediction of 7.4x. A power law "
    "through the venular capillary limb (12.5 um at 0.47 mm/s) and that 3 mm anchor gives "
    "v ~ d^0.67, against d^0.93 arterially, which is physically expected: the venous bed holds more "
    "blood at lower velocity for the same flow. That yields 1.9 mm/s at 100 um, 3.9 at 300 um and "
    "5.5 at 500 um; the interval spans the measured IQR propagated through the same law. "
    "Consequence: at these velocities inflow enhancement is 1.0-1.3x, so TOF of cutaneous veins is "
    "effectively a steady-state spoiled GRE", log=True)

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
    "venous_vessel_velocity": VENOUS_VESSEL_VELOCITY,
    "thermoregulatory_factor": THERMOREGULATORY_FACTOR,
    "target_diameter": TARGET_DIAMETER,
    "vessel_angle_to_slab_normal": VESSEL_ANGLE_TO_SLAB_NORMAL,
}

# Uncertain quantities set by the hardware and the imaging situation rather
# than by tissue composition, ranged and sampled like everything else.
ACQUISITION: dict[str, Param] = {
    "background_structure_rel": Param(
        0.02, 0.20, 0.08, VL,
        "spatial structure of the background as a fraction OF THAT BACKGROUND'S OWN MEAN SIGNAL: "
        "fibrous septa, the dermis/fat boundary, hair follicles, and partial volume between them. "
        "Relative to its own mean is the only definition that stays physical, because the absolute "
        "texture a sequence has to beat is this fraction times the background the sequence produces: "
        "suppress the fat and its texture goes with it. Quoted instead against a fixed reference "
        "signal, 6 % would exceed the whole suppressed-fat background by 2.5x, which is impossible. "
        "The absolute figure detect.py needs is therefore derived, in predict.Prediction, never stored. "
        "Texture does NOT average down as sensitivity improves, so it sets the ceiling on "
        "contrast-to-texture: peak-voxel CNR tends to f |contrast| / (rel x background) as SNR grows "
        "rather than growing without bound, which is what decides whether the regime is "
        "sensitivity-limited or texture-limited and whether a finer voxel can ever pay. No measurement "
        "for skin at these voxel sizes was located and none is derivable here, so this is a SWEPT AXIS, "
        "not an estimate: the interval runs from an essentially smooth background (2 %) to one whose "
        "septa and boundaries modulate it by a fifth (20 %). Estimating it from a background region of "
        "an image belongs in measured.py, never here"),
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
