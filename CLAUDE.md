# CLAUDE.md — design decisions

## Purpose

A Bloch simulation toolbox that predicts what the Siemens `fl3d1r` TOF product sequence can detect: subcutaneous vessels at 1–10 mm depth in the lower leg, at 3 T (Prisma, body transmit, 4-channel Special Purpose surface receive coil). Part of the SKIN-COILS project. Target: ISMRM 2027 abstract, deadline 28 October 2026.

The questions the toolbox must answer are in [QUESTIONS.md](QUESTIONS.md). That list is the stopping criterion.

## The hard constraint: predictions are never fitted to the in vivo data

The abstract's argument is *simulation predicts → images are consistent with it*. That argument is worthless if the model was tuned on the images.

- **Allowed as input:** acquisition settings, which are protocol parameters, not observations. TR, TE, flip angle, bandwidth, voxel size, slab thickness and orientation, TONE percentage, scan time, number of receive channels, acceleration factor.
- **Not allowed as input:** any measured signal, SNR, noise level, contrast ratio, vessel calibre, apparent width or background texture level from the 8 or 13 October 2026 sessions.
- Tissue parameters come from the literature with honest confidence ratings. Free parameters (`kappa`, saturation factors, `structure_fraction`) are set from first principles or literature and carry intervals; none is tuned to make a prediction match.
- Where a quantity cannot be set without a measurement, it gets a wide interval and a docstring saying so. It never quietly adopts the measured value.
- Every prediction must be reproducible by someone who has never seen the images.
- Validation against the measurements is a separate, later task (`measured.py`, `compare.py`, and a `PREDICTIONS` record committed before any comparison). Hooks for it are arguments, never stored numbers: `noise.sigma_from_background`, `structure_fraction`, `snr_available`. No test asserts agreement with the October measurements.

## Scope

Everything is acquired with `fl3d1r`. No `.seq` file runs on the scanner for this abstract. The reachable knobs, and the only acquisition choices the toolbox models, are: flip angle, TR, TE, fat saturation on/off, TONE ramp percentage, MTC on/off, travelling saturation band position, slab thickness and orientation, voxel size.

- **TOF** is the subject. Its signal model is a spoiled GRE with inflow, laminar shells, TONE, and non-selective saturation pulses (fat saturation, MTC) applied every TR.
- **T2-prep** is retained as a *comparison calculation*, not an arm to be acquired: it gives a defensible number for what a flow-independent alternative would deliver at these tissue parameters and what it would cost in scan time.

Out of scope: k-space encoding and image synthesis. The point-spread function enters only as a stated width (`detectability.apparent_fwhm`).

## The regime: sensitivity-limited, not contrast-limited

With fat saturation on, the background at 1–10 mm depth is suppressed fat, close to the noise floor at these voxel sizes. A contrast *ratio* against a near-zero background is not a meaningful quantity.

- **The metric is absolute CNR per √(scan time):** (S_blood − S_background) / σ, with σ set by voxel volume, scan time and g-factor. The ratio is never optimised.
- **Noise is non-central chi with 2N degrees of freedom** for a sum-of-squares combination of N channels, not Rician (`noise.py`).
- **Detection uses the Rose criterion (CNR ≥ 4)** on the peak voxel, with partial volume, grid registration (`kappa`, derived by simulating sub-voxel offsets) and background texture that does not average down (`structure_fraction`).

## Backgrounds

Every contrast result must state which background tissue it refers to. Most target-calibre vessels lie in the hypodermis, so fat — suppressed or not — is at least as important a reference as dermis. Results quoted against only one are incomplete.

## Geometry — the decision that matters most

Slab orientation and vessel orientation are separate parameters and must never be conflated.

- **Slab orientation** is the angle between the slab normal and the skin surface normal (0° = slab parallel to the skin; a transversal slab on the leg, normal along the leg axis, is 90°).
- **Vessel orientation** is the vessel's own direction in tissue (polar angle from the skin normal, azimuth from the slab tilt direction).
- **Path length** through the excited volume follows from both.
- **Background tissue** follows from depth, independently of either.
- **Through-slab position** ζ runs from 0 to 1 along the slab normal. The TONE ramp and the travelling saturation band are defined in this coordinate; a vessel parallel to the slab stays at one ζ.

## No hidden constants

Any geometric cap or assumed timing is either derived from stated inputs or exposed as a swept parameter with a comment saying it is unjustified. TR, TE and time per k-space line are computed from named components, never assumed.

## Parameters and uncertainty

- Tissue properties are intervals with a confidence rating and a source, never point values. `T2*` never exceeds `T2`: every `T2*` interval is capped at its tissue's `T2` lower bound.
- Uncertainty is propagated by Latin-hypercube sampling over the intervals (`tissues.sample`), with a deterministic seed recorded in every output. Figures show bands over the intervals, never single nominal curves.
- `TissueSample` fields may be arrays along a sample axis; sequence code vectorises over that axis and keeps pulse loops sequential.

## Conventions

- Units: ms, mm, mm/s; Hz for bandwidth and off-resonance.
- Angles: degrees at the API boundary, radians inside.
- Public functions under 20 lines, each with a one-sentence docstring.
- Tests: pytest. No test, no merge.
- Figure scripts read every parameter from `protocols.py` and `tissues.py`, never from literals.

## Prior results — hypotheses, not facts

These come from the earlier codebase (SkinMRA) and were computed with placeholder tissue parameters. Reproduce or contradict them; never assume them.

1. Inflow contrast fades along the vessel with visible length ≈ v·T_sat, with T_sat ≈ 0.81 s. (SkinSim's first TOF look, 7 October, found visible length ∝ v but with a T_sat that depends on threshold and background.)
2. Tilting the slab away from the skin surface gains at most ~36 % and costs ~5× the number of slabs.

## Superseded decisions

Recorded so they are not re-litigated by accident. Everything removed is recoverable from the tag `pre-tof-refactor`.

- **Mechanism comparison as the purpose** (TOF vs PD vs T2-prep vs IR vs bSSFP), and the six questions that served it. Replaced on 14 October 2026, after the in vivo sessions of 8 and 13 October fixed the scope to the `fl3d1r` product sequence.
- **Inversion recovery** (`sequences/ir.py`, IR tests): never acquired and will not be. Removed.
- **bSSFP**: never built, never acquired. Dropped.
- **FSD** (flow-sensitive dephasing): not reachable on the product sequence. The spatial position carried in `bloch.SpinState` stays but nothing plans to use it.
- **MT as a two-pool model**: replaced by an empirical per-tissue saturation factor for MTC (`sequences/prep.py`), the honest level of detail for a quantity whose only planned measurement is one on/off comparison.
- **The contrast ratio as the figure of merit**: replaced by absolute CNR per √(scan time).
- **`figures/fig_tof_profile.py`**: replaced by `figures/fig_through_slab.py`.
- **`predictions.py`**: predicted the 8 October protocol set, including IR and T2-prep arms that were never acquired. Removed. `PREDICTIONS.md` is kept unchanged as the timestamped record it produced.
- **Prior results on mechanism ranking** from SkinMRA — the T2-prep/TOF velocity sensitivity and crossover, the blood/dermis vs blood/fat T2 and T1 ratios, bSSFP fat brightness, and the saturated blood/dermis ratio — answered questions that are no longer asked.
