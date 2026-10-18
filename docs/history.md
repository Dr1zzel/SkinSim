# Superseded decisions and archived results

Nothing in this file describes the current toolbox. It records choices that were
made and then abandoned, and results that were computed and then superseded, so
that they are not accidentally re-litigated or mistaken for findings.

For what the toolbox does now, see [README.md](../README.md); for the decisions
that still hold, [CLAUDE.md](../CLAUDE.md).

## Prior results — hypotheses, not facts

These came from the earlier codebase (SkinMRA) and were computed with
placeholder tissue parameters. They were to be reproduced or contradicted, never
assumed.

1. Inflow contrast fades along the vessel with visible length ≈ v·T_sat, with
   T_sat ≈ 0.81 s. SkinSim's first TOF look (7 October 2026) found visible
   length ∝ v but with a T_sat that depends on threshold and background.
2. Tilting the slab away from the skin surface gains at most ~36 % and costs
   ~5× the number of slabs.

## SkinMRA is archived

The earlier codebase (`../SkinMRA`) was superseded and archived on 14 October
2026.

Ported from it: the Latin-hypercube/Sobol sampler (`tissues.collect_params`,
`tissues.sample`, now returning `TissueSample` via `tissues.sample_tissues`);
`signal_weighting` (PD · exp(−TE/T2*)); `cnr_efficiency` (`analysis.py`, against
absolute CNR); the haemodynamic parameters; and the literature validation of
inflow enhancement (`tests/test_validation.py`).

Deliberately not ported: the PyPulseq block-walking machinery.

## Superseded decisions

Everything removed is recoverable from the tag `pre-tof-refactor`.

- **Mechanism comparison as the purpose** (TOF vs PD vs T2-prep vs IR vs bSSFP),
  and the six questions that served it. Replaced on 14 October 2026, after the
  in vivo sessions of 8 and 13 October fixed the scope to the `fl3d1r` product
  sequence.
- **Inversion recovery** (`sequences/ir.py`, IR tests): never acquired and will
  not be. Removed.
- **bSSFP**: never built, never acquired. Dropped.
- **FSD** (flow-sensitive dephasing): not reachable on the product sequence. The
  spatial position carried in `bloch.SpinState` stays but nothing uses it.
- **MT as a two-pool model**: replaced by an empirical per-tissue saturation
  factor for MTC (`sequences/prep.py`), the honest level of detail for a
  quantity whose only planned measurement is one on/off comparison.
- **The contrast ratio as the figure of merit**: replaced by absolute CNR per
  √(scan time).
- **`figures/fig_tof_profile.py`**: replaced by `figures/fig_through_slab.py`.
- **`predictions.py`**: predicted the 8 October protocol set, including IR and
  T2-prep arms that were never acquired. Removed. `PREDICTIONS.md` is kept
  unchanged as the timestamped record it produced.
- **Prior results on mechanism ranking** from SkinMRA — the T2-prep/TOF velocity
  sensitivity and crossover, the blood/dermis vs blood/fat T2 and T1 ratios,
  bSSFP fat brightness, and the saturated blood/dermis ratio — answered
  questions that are no longer asked.

## Superseded figure scripts

Removed on 18 October 2026, when three scripts were found to produce the same
regime figure and three to produce the same detectability figure. Recoverable
from git history.

| removed | superseded by | why |
|---|---|---|
| `fig_abs1_regime.py` | `fig1b_regime.py` | first draft of the Figure 1b regime panel |
| `fig_general_1_regime.py` | `fig1b_regime.py` | same content as a dimensionless collapse; harder to read |
| `fig_abstract_1_protocol.py` | — | tied to the 13 October operating point; never used in the abstract |
| `fig_abstract_2_detectability.py` | `fig5_detectability.py` | operating-point version of the same panel |

Renamed at the same time, so that a script's name says which figure it makes:
`fig_abs5_detect.py` → `fig5_detectability.py`,
`fig_general_2_tradeoffs.py` → `fig_tradeoffs.py`,
`fig_general_3_tr_flip.py` → `fig_tr_flip_ridge.py`.

## Corrections worth remembering

- **Target vessel velocity** was revised from 5–50 mm/s up to 10–150 mm/s on
  14 October on anatomical grounds, then restored to 5–50 on 16 October: the
  revised entry had been written for 0.3–1 mm vessels and was being applied to
  the 100–500 µm target, a calibre band smaller. The restored interval has a
  derived centre (v ∝ d between the capillaroscopy and perforator anchors) and a
  deliberately wider span.
- **Blood MT** was recorded as "bound-pool fraction < 1 %, so barely saturated"
  until Stanisz et al. 2005 was read in full on 18 October: it measures
  2.8 ± 0.7 % with MTR 11 ± 4 %. Muscle was recorded as "MTR ~20–40 %" against a
  measured 88 ± 2 %.
- **Subcutaneous fat T1 and muscle T1** carried MEDIUM intervals 40 and 50 ms
  wide, set from single sources, against literature spreads of 253–450 ms and
  898–1509 ms across methods.
