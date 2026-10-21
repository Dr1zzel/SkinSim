# SkinSim

A Bloch simulation toolbox for **time-of-flight MR angiography of cutaneous
vessels at 3 T**: can a spoiled-GRE TOF acquisition see subcutaneous vessels of
100–500 µm lying 1–10 mm below the skin surface, and with what protocol?

Built for the [SKIN-COILS](https://anr.fr) project (ANR/FWF), Medical
University of Vienna and Université Paris-Saclay.

---

## The model in one paragraph

Under ideal spoiling the only magnetisation that survives from one TR to the
next is the longitudinal component `Mz`, and every event acting on it —
relaxation, an excitation, a saturation pulse — is an **affine map**
`Mz → A·Mz + B`. Composing the events of one TR gives a single map. Static
tissue sits at that map's fixed point `B/(1−A)`; blood that has received `n`
excitations is the map applied `n` times, available in closed form. Spins are
followed **along the vessel** rather than within a fixed voxel, because inflow
and outflow are exactly what an EPG description cannot express. Flow is laminar,
resolved into annular shells, so near-wall blood saturates differently from the
centreline.

The whole engine is `affine.py` — 58 lines, no MR-specific code in it.

## Install and run

Requires Python 3.11+, NumPy, SciPy and Matplotlib.

```bash
pip install numpy scipy matplotlib pytest
make test          # the whole suite
make figures       # regenerate every figure into figures/
make check         # tests, then verify the committed figures match the code
```

To understand the pipeline, run the trace — it re-derives one figure stage by
stage and prints every intermediate:

```bash
python figures/trace_fig1b.py --draw 0 --velocity 20
```

and read [docs/figure1b-walkthrough.md](docs/figure1b-walkthrough.md) beside it.

## What the modules do

| module | contents |
|---|---|
| `affine.py` | the engine: affine maps on `Mz`, composition, powers, fixed points |
| `tissues.py` | every parameter as an interval with a confidence rating and a source, plus the Latin-hypercube sampler |
| `sequence.py` | readout timing, saturation pulses, TONE, and the TOF signal model |
| `geometry.py` | skin layers, vessel geometry, laminar shells, the slab and the path through it |
| `protocols.py` | TR and TE assembled from named timing components, and the named comparisons the figures draw |
| `detect.py` | channel noise statistics, CNR per √(scan time), partial volume, apparent width, SNR required to detect |
| `predict.py` | the composition layer: protocol + vessel + tissues → contrast, CNR efficiency and a detection verdict |

The stack runs one way. `affine` knows no MR; `tissues`, `geometry`, `sequence`
and `protocols` build the signal model; `detect` is pure statistics and
geometry and imports nothing from the project. Only `predict` joins the last
two, and a test asserts that nothing below it imports it.

Tests mirror the modules and are run by `make test`. `tests/reference.py` holds independent closed-form implementations used to check the fast path; nothing in the toolbox imports it, which is what makes the comparison a real check.

## Figures

Scripts that produce figures for the abstract:

| script | figure |
|---|---|
| `figures/fig1a_anatomy.py` | **Figure 1a** — the cutaneous vascular tree to scale, against the target calibre |
| `figures/fig1b_regime.py` | **Figure 1b** — inflow against the backgrounds it competes with |
| `figures/fig4_efficiency.py` | **Figure 4** — CNR efficiency over velocity × orientation, by contrast mechanism and flip |
| `figures/fig5_detectability.py` | **Figure 5** (simulation panel) — what finer voxels cost, and what width they can report |

`figures/trace_fig1b.py` is not a figure: it re-derives Figure 1b stage by stage
and prints every intermediate, as the companion to the walkthrough.

Every figure records its Latin-hypercube seed and draw count in its own footer,
so any panel can be regenerated exactly.

## The constraint that matters

**No prediction is fitted to the in vivo data.** The argument the abstract makes
is *simulation predicts → images are consistent with it*, which is worth nothing
if the model was tuned on the images.

- **Allowed as input:** acquisition settings — TR, TE, flip angle, bandwidth,
  voxel size, slab thickness and orientation, TONE percentage, scan time,
  channel count, acceleration factor. These are protocol parameters, not
  observations.
- **Not allowed:** any measured signal, SNR, noise level, contrast ratio, vessel
  calibre, apparent width or background texture from this project's sessions.

Tissue parameters come from the literature with confidence ratings and sources.
Where a quantity cannot be set without a measurement it gets a wide interval and
says so, rather than quietly adopting a measured value. Every prediction is
reproducible by someone who has never seen the images.

## Honesty about the parameters

`tissues.py` marks each source `[checked]` when it has been read at the
publisher and `[unchecked]` when it is a lead that has not been consulted.

```python
from tissues import low_confidence_report, unchecked_sources
print(low_confidence_report())   # every LOW and VERY_LOW parameter, with its source
print(unchecked_sources())       # sources not yet read at the publisher
```

Most parameters are LOW or VERY_LOW. Skin relaxation times come from a 1.5 T
study (Richard et al. 1991) and are shifted to 3 T rather than measured there;
dermis `fatsat_s` and `mtc_s` rest on physical arguments rather than
measurements, and epidermis T2* has no measurement behind it at all. This is deliberate — the uncertainty analysis is the
point, not a robustness appendix — but it should be read before any number here
is quoted.

## Further reading

- [CLAUDE.md](CLAUDE.md) — design decisions that still hold
- [docs/figure1b-walkthrough.md](docs/figure1b-walkthrough.md) — one figure
  traced through the whole pipeline
- [docs/history.md](docs/history.md) — superseded decisions and corrections
- [QUESTIONS.md](QUESTIONS.md) — the questions the toolbox must answer
