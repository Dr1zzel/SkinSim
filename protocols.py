"""Acquisition protocols: the 13 October 2026 operating point and candidates not yet run.

ACQUISITION SETTINGS ONLY. Everything here is a protocol parameter decoded
from the embedded Siemens MrPhoenixProtocol (not from the series names, several
of which are wrong) or a stated hardware specification. No measured signal,
SNR, noise, contrast, calibre or width enters this module (CLAUDE.md).

Measured and candidate protocols are both plain Protocol objects; the ones
already acquired are not privileged in the API.

Timing. The console reports TR 45.22 ms and TE 5.13 ms; the decoded settings
give the readout bandwidth but not the RF duration or gradient-lobe timings.
fl3d1r_timing therefore derives the readout from the bandwidth and the
encoding lobes from the voxel size at the gradient limits, infers the RF
duration from the console TE (an acquisition setting, so allowed), and leaves
the rest of TR as dead time. TR and TE are reproduced by construction; what
carries information is whether the inferred RF duration is physically
plausible. Replace it with the protocol's RF duration once decoded.
Units: ms, mm, s for scan times, Hz/px for bandwidth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from geometry import SatBand, Slab
from sequence import ReadoutTiming, SaturationModule, TofSequence
from numpy.typing import ArrayLike


GAMMA_HZ_PER_MT = 42577.48  # 1H gyromagnetic ratio / 2 pi


@dataclass(frozen=True)
class GradientSystem:
    """Gradient limits used to derive minimum lobe durations."""

    max_amplitude_mt_per_m: float
    max_slew_mt_per_m_per_ms: float  # T/m/s = mT/m/ms


# Siemens Prisma XR specification. Assumes the protocol runs in the fastest
# gradient mode; a softer mode lengthens every lobe. UNVERIFIED: read the
# gradient mode from MrPhoenixProtocol.
PRISMA_XR = GradientSystem(max_amplitude_mt_per_m=80.0, max_slew_mt_per_m_per_ms=200.0)

# Spoiler dephasing across one partition, in cycles. Assumed, not decoded:
# it only moves time between the spoiler and the dead-time remainder.
SPOILER_CYCLES_PER_PARTITION = 2.0


def encoding_area(pixel_mm: float) -> float:
    """Return the gradient area in mT ms/m that moves k-space from centre to edge for pixel_mm."""
    return 1e6 / (2 * GAMMA_HZ_PER_MT * pixel_mm)  # 1 / (2 gamma dx), mT s/m -> mT ms/m with dx in mm


def min_lobe_ms(area_mt_ms_per_m: float, gradients: GradientSystem) -> float:
    """Return the shortest trapezoid or triangle lobe that delivers the area at the gradient limits."""
    g, s = gradients.max_amplitude_mt_per_m, gradients.max_slew_mt_per_m_per_ms
    if area_mt_ms_per_m <= g * g / s:
        return 2 * math.sqrt(area_mt_ms_per_m / s)
    return area_mt_ms_per_m / g + g / s


def encoding_lobes_ms(inplane_mm: float, partition_mm: float, gradients: GradientSystem) -> tuple[float, float, float]:
    """Return (rephaser, prephaser, spoiler) lobe durations derived from voxel size and gradient limits."""
    rephaser = min_lobe_ms(encoding_area(partition_mm), gradients)  # partition encode; slab rephaser not included
    prephaser = min_lobe_ms(encoding_area(inplane_mm), gradients)  # read prephaser with phase encode alongside
    spoiler = min_lobe_ms(2 * SPOILER_CYCLES_PER_PARTITION * encoding_area(partition_mm), gradients)
    return rephaser, prephaser, spoiler


def infer_rf_duration_ms(te_ms: float, inplane_mm: float, partition_mm: float,
                         bandwidth_per_px_hz: float, gradients: GradientSystem) -> float:
    """Return the RF duration that makes back-to-back lobes reproduce te_ms, refusing a non-positive one."""
    rephaser, prephaser, _ = encoding_lobes_ms(inplane_mm, partition_mm, gradients)
    rf_ms = 2 * (te_ms - 500.0 / bandwidth_per_px_hz - rephaser - prephaser)
    if rf_ms <= 0:
        raise ValueError(f"TE {te_ms} ms is shorter than the back-to-back lobes allow: the component model is wrong")
    return rf_ms


def fl3d1r_timing(inplane_mm: float, partition_mm: float, bandwidth_per_px_hz: float,
                  rf_duration_ms: float, tr_ms: float, gradients: GradientSystem) -> ReadoutTiming:
    """Return the readout timing from derived lobes and a given RF duration, with dead time filling TR."""
    rephaser, prephaser, spoiler = encoding_lobes_ms(inplane_mm, partition_mm, gradients)
    used = rf_duration_ms + rephaser + prephaser + 1000.0 / bandwidth_per_px_hz + spoiler
    if used > tr_ms:
        raise ValueError(f"components need {used:.2f} ms, longer than TR {tr_ms} ms")
    return ReadoutTiming(rf_duration_ms=rf_duration_ms, rephaser_ms=rephaser, prephaser_ms=prephaser,
                         bandwidth_per_px_hz=bandwidth_per_px_hz, spoiler_ms=spoiler, dead_time_ms=tr_ms - used)


@dataclass(frozen=True)
class Protocol:
    """One fl3d1r acquisition: settings as decoded, plus derived voxel, line counts and scan time."""

    name: str
    acquired: str | None  # date acquired, None for a candidate not yet run
    timing: ReadoutTiming
    flip_deg: float
    tone_pct: float
    tone_direction: int  # +1: TONE ramp rises towards the slab's +normal, the declared flow direction
    fat_sat: bool
    mtc: bool
    sat_band: SatBand | None
    slab_orientation_deg: float  # slab normal to skin normal; 90 = transversal on the leg
    n_partitions: int
    partition_mm: float
    slice_oversampling: float  # fraction of extra partitions acquired and discarded
    inplane_mm: float
    read_fov_mm: float
    phase_fov_mm: float
    grappa_r: int
    grappa_ref_lines: int
    partial_fourier_slice: float
    partial_fourier_phase: float
    n_channels: int
    console_ta_s: float | None  # stated by the console; compared with, never substituted for, scan_time_s
    fat_sat_delay_ms: float  # fat-sat pulse centre to excitation. UNJUSTIFIED: sweep 4-15 ms
    mtc_delay_ms: float  # MTC pulse centre to excitation. UNJUSTIFIED: sweep 4-15 ms

    @property
    def slab_thickness_mm(self) -> float:
        """Return the reconstructed slab thickness, partitions times partition thickness."""
        return self.n_partitions * self.partition_mm

    @property
    def voxel_volume_mm3(self) -> float:
        """Return the voxel volume, in-plane squared times partition thickness."""
        return self.inplane_mm**2 * self.partition_mm

    @property
    def lines_per_partition(self) -> int:
        """Return phase-encode lines acquired per partition, GRAPPA reference lines included."""
        n_phase = round(self.phase_fov_mm / self.inplane_mm * self.partial_fourier_phase)
        ref = min(self.grappa_ref_lines, n_phase)
        return math.ceil((n_phase - ref) / self.grappa_r) + ref

    @property
    def acquired_partitions(self) -> int:
        """Return partitions acquired, with slice oversampling and slice partial Fourier."""
        return math.ceil(self.n_partitions * (1 + self.slice_oversampling) * self.partial_fourier_slice)

    @property
    def total_lines(self) -> int:
        """Return every k-space line acquired, one per TR."""
        return self.lines_per_partition * self.acquired_partitions

    @property
    def scan_time_s(self) -> float:
        """Return the derived scan time, total lines times TR."""
        return self.total_lines * self.timing.tr_ms / 1000.0

    def slab(self) -> Slab:
        """Return the excitation slab."""
        return Slab(thickness_mm=self.slab_thickness_mm, orientation_deg=self.slab_orientation_deg)

    def sequence(self, band_pulse_residual: ArrayLike = 0.0) -> TofSequence:
        """Return the TOF signal model for this protocol; pass sampled band residuals, 0 being an ideal 90 deg pulse."""
        modules = [SaturationModule("fatsat_s", self.fat_sat_delay_ms)] if self.fat_sat else []
        modules += [SaturationModule("mtc_s", self.mtc_delay_ms)] if self.mtc else []
        return TofSequence(timing=self.timing, flip_deg=self.flip_deg, tone_pct=self.tone_pct,
                           tone_direction=self.tone_direction, modules=tuple(modules),
                           sat_band=self.sat_band, band_pulse_residual=band_pulse_residual)


# --- 13 October 2026: five series sharing one operating point --------------------

CONSOLE_TR_MS, CONSOLE_TE_MS, BANDWIDTH_PER_PX_HZ = 45.22, 5.13, 186.0
FINEST_INPLANE_MM, PARTITION_MM = 0.2578, 0.23
RF_DURATION_MS = infer_rf_duration_ms(CONSOLE_TE_MS, FINEST_INPLANE_MM, PARTITION_MM, BANDWIDTH_PER_PX_HZ, PRISMA_XR)
# All five series share TR and TE; the finest in-plane resolution sets the lobes.
MEASURED_TIMING = fl3d1r_timing(FINEST_INPLANE_MM, PARTITION_MM, BANDWIDTH_PER_PX_HZ,
                                RF_DURATION_MS, CONSOLE_TR_MS, PRISMA_XR)

_OPERATING_POINT = Protocol(
    name="", acquired="2026-10-13", timing=MEASURED_TIMING, flip_deg=18.0, tone_pct=70.0,
    tone_direction=+1,  # ASSUMED: declared flow direction not decoded; the band (side +1) then sits downstream
    fat_sat=True,
    mtc=False,  # ASSUMED: MTC is not listed in the decoded operating point; confirm from MrPhoenixProtocol
    sat_band=SatBand(thickness_mm=40.0, gap_mm=10.0, side=+1),
    slab_orientation_deg=90.0,  # slab normal along the leg axis
    n_partitions=48, partition_mm=PARTITION_MM, slice_oversampling=0.25, inplane_mm=FINEST_INPLANE_MM,
    read_fov_mm=99.0, phase_fov_mm=55.7, grappa_r=2, grappa_ref_lines=32,
    partial_fourier_slice=7 / 8, partial_fourier_phase=1.0, n_channels=4, console_ta_s=None,
    fat_sat_delay_ms=8.0, mtc_delay_ms=8.0,
)

MEASURED_13OCT: dict[str, Protocol] = {
    p.name: p for p in (
        replace(_OPERATING_POINT, name="13oct_0.26mm", inplane_mm=0.2578, console_ta_s=349.9),
        replace(_OPERATING_POINT, name="13oct_0.35mm", inplane_mm=0.3477, console_ta_s=246.8),
        replace(_OPERATING_POINT, name="13oct_0.46mm", inplane_mm=0.4635, console_ta_s=179.4),
        # ASSUMED: the decoded summary does not say which resolution these two variants used.
        replace(_OPERATING_POINT, name="13oct_0.35mm_band_opposite", inplane_mm=0.3477, console_ta_s=246.8,
                sat_band=SatBand(thickness_mm=40.0, gap_mm=10.0, side=-1)),
        replace(_OPERATING_POINT, name="13oct_0.35mm_slab_parallel_to_skin", inplane_mm=0.3477,
                console_ta_s=246.8, slab_orientation_deg=0.0),
    )
}

# --- 20 October 2026: four arms at 0.22 mm, near-isotropic -----------------------
# Decoded from each series' own embedded MrPhoenixProtocol. A key-by-key diff of
# the four protocols found 19 meaningful differences and no others, so what
# separates the arms is exactly: TR, the saturation modules, and - on the
# reversed arm only - the band geometry.

# KNOWN SHORTFALL: the derived scan time comes out 0.84x the console's for all
# four arms. The protocol declares sKSpace.lPhaseEncodingLines = 293 while
# lines_per_partition derives 244 from the phase FOV and voxel, and 293/244 =
# 1.20 is exactly the missing factor. The acquired line count is evidently not
# the reconstructed matrix. Scan time feeds detect.cnr_efficiency, so this is
# worth fixing with an explicit acquired-lines field rather than a fudge.
OCT20_TE_MS, OCT20_BANDWIDTH_HZ = 5.86, 187.0
OCT20_INPLANE_MM, OCT20_PARTITION_MM = 0.217593, 0.22
OCT20_RF_DURATION_MS = infer_rf_duration_ms(OCT20_TE_MS, OCT20_INPLANE_MM, OCT20_PARTITION_MM,
                                            OCT20_BANDWIDTH_HZ, PRISMA_XR)


def oct20_timing(tr_ms: float) -> ReadoutTiming:
    """Return the 20 October readout timing at one arm's TR; every other component is shared."""
    return fl3d1r_timing(OCT20_INPLANE_MM, OCT20_PARTITION_MM, OCT20_BANDWIDTH_HZ,
                         OCT20_RF_DURATION_MS, tr_ms, PRISMA_XR)


_OCT20 = Protocol(
    name="", acquired="2026-10-20", timing=oct20_timing(50.60), flip_deg=25.0, tone_pct=70.0,
    tone_direction=+1,  # ASSUMED, as on 13 October: the declared flow direction is not decoded
    fat_sat=True, mtc=False,
    sat_band=SatBand(thickness_mm=40.0, gap_mm=10.0, side=+1),
    # The slab normal is 83.2 deg from the head-foot axis, so the slab lies ALONG
    # the limb and is roughly tangent to the skin: orientation 0, not the 90 deg
    # of 13 October, despite "tra" in every series name. Verified two ways - the
    # protocol's own sNormal and the DICOM ImageOrientationPatient agree to 1.0000.
    slab_orientation_deg=0.0,
    n_partitions=52, partition_mm=OCT20_PARTITION_MM, slice_oversampling=0.153846153846,
    inplane_mm=OCT20_INPLANE_MM, read_fov_mm=94.0, phase_fov_mm=53.0925925926,
    grappa_r=2, grappa_ref_lines=32, partial_fourier_slice=7 / 8, partial_fourier_phase=1.0,
    n_channels=4, console_ta_s=442.0, fat_sat_delay_ms=8.0, mtc_delay_ms=8.0,
)

MEASURED_20OCT: dict[str, Protocol] = {
    p.name: p for p in (
        replace(_OCT20, name="20oct_fatsat"),
        replace(_OCT20, name="20oct_nosat", timing=oct20_timing(35.06), fat_sat=False,
                console_ta_s=307.0),
        # ucTOFFlowInNormalDir is set on this arm alone, which reverses the declared
        # flow direction, so the TONE ramp and the band side both flip. ASSUMED that
        # the one flag drives both. The band itself also changed, which makes this a
        # weaker, closer band on the other side rather than the same band mirrored.
        replace(_OCT20, name="20oct_band_reversed", tone_direction=-1,
                sat_band=SatBand(thickness_mm=10.0, gap_mm=2.0, side=-1)),
        replace(_OCT20, name="20oct_fatsat_mtc", timing=oct20_timing(63.87), mtc=True,
                console_ta_s=556.0),
    )
}

MEASURED: dict[str, Protocol] = {**MEASURED_13OCT, **MEASURED_20OCT}

# --- Candidates not yet run ----------------------------------------------------------

CANDIDATE_BASE = MEASURED["13oct_0.35mm"]
CANDIDATE_FLIPS_DEG = (8.0, 12.0, 15.0, 22.0, 26.0, 30.0)


def candidate(base: Protocol, name: str, **changes) -> Protocol:
    """Return a protocol not yet acquired, derived from base by the given setting changes."""
    return replace(base, name=name, acquired=None, console_ta_s=None, **changes)


CANDIDATES: dict[str, Protocol] = {
    p.name: p for p in (
        candidate(CANDIDATE_BASE, "fatsat_off", fat_sat=False),  # same TR and lines: matched scan time
        candidate(CANDIDATE_BASE, "tone_off", tone_pct=0.0),
        candidate(CANDIDATE_BASE, "no_acceleration", grappa_r=1, grappa_ref_lines=0),
        candidate(CANDIDATE_BASE, "inplane_0.18mm", inplane_mm=0.18,
                  timing=fl3d1r_timing(0.18, PARTITION_MM, BANDWIDTH_PER_PX_HZ, RF_DURATION_MS,
                                       CONSOLE_TR_MS, PRISMA_XR)),
        *(candidate(CANDIDATE_BASE, f"flip_{f:g}deg", flip_deg=f) for f in CANDIDATE_FLIPS_DEG),
    )
}

ALL_PROTOCOLS: dict[str, Protocol] = {**MEASURED, **CANDIDATES}

# --- Named comparisons the figures draw ----------------------------------------------
# Kept here rather than in the figure scripts so a protocol change cannot leave a
# figure quoting a resolution the toolbox no longer models.

# The resolution ladder: the three in-plane voxels acquired on 13 October and the
# finer candidate. Keys index ALL_PROTOCOLS; values label the axis.
RESOLUTION_SWEEP: dict[str, str] = {
    "inplane_0.18mm": "0.18 mm (candidate)",
    "13oct_0.26mm": "0.26 mm",
    "13oct_0.35mm": "0.35 mm",
    "13oct_0.46mm": "0.46 mm",
}

# The sensitivity unit for relative comparisons: absolute SNR needs the coil's
# sensitivity at depth, which is not derivable here, so figures express every
# protocol in units of what this one delivers (detect.thermal_snr).
REFERENCE_PROTOCOL: Protocol = MEASURED["13oct_0.26mm"]

# The contrast mechanisms to compare at fixed geometry: what each arm changes
# about the background it has to beat.
CONTRAST_ARMS: dict[str, str] = {
    "13oct_0.35mm": "fat saturation (operating point)",
    "fatsat_off": "no fat saturation",
    "tone_off": "fat saturation, TONE off",
    "13oct_0.35mm_band_opposite": "fat saturation, band upstream",
}
