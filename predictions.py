"""Predicted blood/background contrast for the acquisition protocols, written to PREDICTIONS.md.

Arterial blood, all tissue values nominal. bSSFP is skipped (no sequence yet).
Commit PREDICTIONS.md before acquiring: its timestamp and commit hash are the record.
Run from the repository root: python predictions.py
"""
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from acquisition import Slab
from sequences.base import FlowGeometry, ReadoutTiming, Sequence, TissueSample
from sequences.ir import IrSequence, nulling_ti, textbook_nulling_ti
from sequences.t2prep import T2PrepSequence
from sequences.tof import TofSequence
from skin import Vessel, tissue_at_depth
from tissues import ARTERIAL_BLOOD, HYPODERMIS, Tissue

ROOT = Path(__file__).resolve().parent
TIMING = ReadoutTiming(rf_duration_ms=2.0, rephaser_ms=1.0, prephaser_ms=1.45,
                       bandwidth_per_px_hz=200.0, spoiler_ms=2.0,
                       dead_time_ms=28.89)  # unmodelled remainder to match console TR 40.34
FLIP_DEG, PD_FLIP_DEG = 18.0, 5.0
SLAB = Slab(thickness_mm=11.52, orientation_deg=90.0)  # transversal: normal along the limb, in the skin plane
VESSEL_POLAR_DEG, VESSEL_AZIMUTH_DEG = 90.0, 0.0  # along the limb, i.e. parallel to the slab normal
VESSEL_DIAMETER_MM = 0.3  # not specified; enters no current calculation
DEPTHS_MM = (1.0, 5.0)
MEAN_VELOCITIES_MM_S = (1.0, 2.5, 7.5, 25.0)
N_SHELLS = 8
MAX_PATH_MM = 30.0  # no justification; must be swept
POSITIONS_MM = np.linspace(0.0, SLAB.thickness_mm, 100)
IR_TI_BRACKET_MS = (150.0, 200.0, 250.0, 300.0)
T2PREP_MS = (40.0, 60.0, 80.0, 100.0)
SEGMENT = dict(lines_per_train=64, recovery_ms=0.0, center_line=0)
BLOOD, FAT = TissueSample.nominal(ARTERIAL_BLOOD), TissueSample.nominal(HYPODERMIS)


def ir(ti_ms: float) -> IrSequence:
    """Return the IR protocol at ti_ms."""
    return IrSequence(timing=TIMING, flip_deg=FLIP_DEG, ti_ms=ti_ms, inversion_efficiency=1.0, **SEGMENT)


def protocols() -> list[tuple[str, Sequence]]:
    """Return every protocol variant to predict, labelled."""
    null_ti = nulling_ti(FAT, ir(0.0))
    variants: list[tuple[str, Sequence]] = [
        ("1. TOF", TofSequence(timing=TIMING, flip_deg=FLIP_DEG)),
        ("2. PD", TofSequence(timing=TIMING, flip_deg=PD_FLIP_DEG)),
        (f"3. IR, TI {null_ti:.1f} ms (nulls fat)", ir(null_ti)),
    ]
    variants += [(f"3. IR, TI {ti:g} ms", ir(ti)) for ti in IR_TI_BRACKET_MS]
    variants += [(f"4. T2-prep, {p:g} ms ({p / 4:g} ms refocusing interval)",
                  T2PrepSequence(timing=TIMING, flip_deg=FLIP_DEG, prep_ms=p, n_refocus=4,
                                 crusher_ms=5.0, **SEGMENT)) for p in T2PREP_MS]
    return variants


def flow(depth_mm: float, velocity_mm_s: float) -> FlowGeometry:
    """Return the vessel-in-slab geometry at depth_mm."""
    vessel = Vessel(depth_mm=depth_mm, polar_deg=VESSEL_POLAR_DEG, azimuth_deg=VESSEL_AZIMUTH_DEG,
                    diameter_mm=VESSEL_DIAMETER_MM)
    return FlowGeometry(vessel, SLAB, velocity_mm_s, N_SHELLS, MAX_PATH_MM)


def ratio_summary(seq: Sequence, depth_mm: float, velocity_mm_s: float) -> tuple[float, float, float]:
    """Return blood/background ratio at slab entry, averaged along the slab, and at slab exit."""
    background = TissueSample.nominal(tissue_at_depth(depth_mm))
    ratio = seq.contrast(BLOOD, background, flow(depth_mm, velocity_mm_s), POSITIONS_MM).ratio
    return float(ratio[0]), float(ratio.mean()), float(ratio[-1])


def fmt(value: float) -> str:
    """Return a ratio or signal formatted to stay readable when a background is nulled."""
    return f"{value:.3f}" if abs(value) < 1000 else f"{value:.2e}"


def signal_table(variants: list[tuple[str, Sequence]]) -> list[str]:
    """Return a markdown table of absolute signals: static backgrounds and blood averaged along the slab."""
    lines = ["| protocol | dermis | fat | blood static | blood entry | "
             + " | ".join(f"blood mean, {v:g} mm/s" for v in MEAN_VELOCITIES_MM_S) + " |",
             "|---|" + "---|" * (4 + len(MEAN_VELOCITIES_MM_S))]
    for label, seq in variants:
        statics = [seq.static_signal(TissueSample.nominal(tissue_at_depth(d))) for d in DEPTHS_MM]
        blood = [seq.blood_signal(BLOOD, flow(DEPTHS_MM[0], v), POSITIONS_MM) for v in MEAN_VELOCITIES_MM_S]
        cells = [*statics, seq.static_signal(BLOOD), blood[0][0], *(b.mean() for b in blood)]
        lines.append(f"| {label} | " + " | ".join(f"{c:.4f}" for c in cells) + " |")
    return lines + [""]


def protocol_table(label: str, seq: Sequence) -> list[str]:
    """Return a markdown table of ratios per velocity and background for one protocol."""
    names = [tissue_at_depth(d).name for d in DEPTHS_MM]
    lines = [f"### {label}", "", f"Time per line {seq.time_per_line_ms():.2f} ms.", "",
             "| mean v (mm/s) | " + " | ".join(f"{n}: entry / mean / exit" for n in names) + " |",
             "|---|" + "---|" * len(names)]
    for v in (*MEAN_VELOCITIES_MM_S, 0.0):
        cells = [" / ".join(fmt(r) for r in ratio_summary(seq, d, v)) for d in DEPTHS_MM]
        lines.append(f"| {v:g}{' (static)' if v == 0 else ''} | " + " | ".join(cells) + " |")
    return lines + [""]


def git_state() -> str:
    """Return the current commit and whether the working tree has uncommitted changes."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = "with uncommitted changes" if git("status", "--porcelain", "--", ".", ":!PREDICTIONS.md") else "clean"
    return f"{git('rev-parse', '--short', 'HEAD')} ({dirty})"


def tissue_lines(tissues: list[Tissue]) -> list[str]:
    """Return the nominal values used, with their confidence."""
    lines = ["| tissue | T1 ms | T2 ms | T2* ms | PD |", "|---|---|---|---|---|"]
    for t in tissues:
        cells = [f"{p.nominal:g} ({p.confidence.name})" for p in (t.t1_ms, t.t2_ms, t.t2star_ms, t.pd_rel)]
        lines.append(f"| {t.name} | " + " | ".join(cells) + " |")
    return lines + [""]


def header() -> list[str]:
    """Return the record header: when, from which code, with which inputs and assumptions."""
    null_ti = nulling_ti(FAT, ir(0.0))
    return [
        "# Predictions", "",
        f"Generated {datetime.now().astimezone().isoformat(timespec='seconds')} "
        f"from commit {git_state()} by `predictions.py`.", "",
        "Blood/background signal ratio, arterial blood, all tissue values nominal. "
        "Each cell gives the ratio at slab entry / averaged over 100 points along the slab / at slab exit. "
        "'static' is the v -> 0 limit.", "",
        f"- Readout: TR {TIMING.tr_ms:.2f} ms, TE {TIMING.te_ms:.2f} ms, flip {FLIP_DEG:g} deg "
        f"(PD {PD_FLIP_DEG:g} deg); dead time {TIMING.dead_time_ms:g} ms is an unmodelled remainder.",
        f"- Geometry: slab {SLAB.thickness_mm:g} mm, transversal (orientation {SLAB.orientation_deg:g} deg); "
        f"vessel along the limb (polar {VESSEL_POLAR_DEG:g}, azimuth {VESSEL_AZIMUTH_DEG:g} deg), "
        f"path {flow(1.0, 1.0).path_mm():.2f} mm; depths {DEPTHS_MM} mm; "
        f"{N_SHELLS} laminar shells; max_path_mm {MAX_PATH_MM:g} (unjustified, inactive here).",
        "- Segmented protocols: 64 lines, recovery 0 ms, centric; preparation non-selective, "
        "so inflowing blood arrives prepared but unsaturated.",
        f"- IR fat null: {null_ti:.1f} ms segmented vs {textbook_nulling_ti(FAT):.1f} ms for T1 ln 2 "
        f"({null_ti - textbook_nulling_ti(FAT):+.1f} ms).",
        "- T2-prep uses one blood T2 for every refocusing interval; Zhao (MRM 2007) shows it varies, "
        "so these predictions do not yet carry that dependence.", "",
    ]


def main() -> None:
    """Print the predictions and write PREDICTIONS.md."""
    lines = header() + ["## Tissue values used", ""]
    lines += tissue_lines([ARTERIAL_BLOOD, *(tissue_at_depth(d) for d in DEPTHS_MM)])
    variants = protocols()
    lines += ["## Predicted signals", "",
              "Echo signal in units of fully relaxed dermis magnetisation (proton density 1). "
              "Blood does not depend on depth; 'blood mean' averages along the slab. "
              "Compare these, not ratios, wherever a background is nulled: a measured ratio "
              "against a nulled background is set by noise and imperfect nulling.", ""]
    lines += signal_table(variants)
    lines += ["## Predicted blood/background ratio", ""]
    for label, seq in variants:
        lines += protocol_table(label, seq)
    text = "\n".join(lines)
    print(text)
    (ROOT / "PREDICTIONS.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
