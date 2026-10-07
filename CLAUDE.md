# CLAUDE.md — design decisions

## Purpose

A Bloch simulation toolbox for choosing an MR contrast mechanism for 100–500 µm subcutaneous vessels at 1–10 mm depth, at 3 T. Part of the SKIN-COILS project. Target: ISMRM 2027 abstract, deadline 28 October 2026.

The questions the toolbox must answer are in [QUESTIONS.md](QUESTIONS.md). That list is the stopping criterion.

## Sequences in scope

- **TOF** (inflow).
- **PD baseline** — spoiled GRE, shortest TE, no preparation. The zero-cost reference every other mechanism is compared against.
- **T2-prep.**
- **Inversion recovery** with TI nulling fat.
- **bSSFP.**

Deferred but not precluded: **FSD** and **MT**.

- Carry a spatial position in the spin state from day one, even though nothing uses it yet (needed for FSD).
- Structure tissues so a bound pool can be added later (needed for MT).

## Out of scope

k-space encoding, image synthesis.

## Backgrounds

Every contrast result must state which background tissue it refers to. Most target-calibre vessels lie in the hypodermis, so fat is at least as important a reference as dermis. Results quoted against only one background are incomplete.

## Geometry — the decision that matters most

Slab orientation and vessel orientation are separate parameters and must never be conflated.

- **Slab orientation** is the angle between the slab normal and the skin surface normal (0° = slab parallel to the skin).
- **Vessel orientation** is the vessel's own direction in tissue. The cutaneous plexus is predominantly in-plane with no preferred azimuth.
- **Path length** through the excited volume follows from both.
- **Background tissue** follows from depth, independently of either.

## No hidden constants

Any geometric cap or assumed timing is either derived from stated inputs or exposed as a swept parameter with a comment saying it is unjustified. TR, TE and time per k-space line are computed from named components, never assumed.

## Parameters

Tissue properties are intervals with a confidence rating and a source, never point values.

## Conventions

- Units: ms, mm, mm/s.
- Angles: degrees at the API boundary, radians inside.
- Public functions under 20 lines, each with a one-sentence docstring.
- Tests: pytest. No test, no merge.

## Prior results — hypotheses, not facts

These come from the earlier codebase and were all computed with placeholder tissue parameters. Reproduce or contradict them; never assume them.

1. Inflow contrast fades along the vessel with visible length ≈ v·T_sat, with T_sat ≈ 0.81 s.
2. The saturated blood/dermis ratio is undetermined: 0.81–1.57 over the parameter ranges, with dermis T2* the dominant term.
3. T2-prep blood signal varies 12–36 % with velocity, against TOF's 350–470 %.
4. The TOF/T2-prep crossover sits at 9–26 mm/s against dermis.
5. Tilting the slab away from the skin surface gains at most ~36 % and costs ~5× the number of slabs.
6. The separating tissue property appears to differ by background: blood/dermis T2 ratio 3.7 against T1 ratio 1.4, but blood/fat T2 ratio 1.3 against T1 ratio 4.5. If so, the two layers favour different mechanisms. Test this rather than assuming it.
7. bSSFP signal ∝ √(T2/T1) makes fat the brightest tissue (blood/fat ≈ 0.49 unsuppressed), so fat suppression is load-bearing for that arm.
