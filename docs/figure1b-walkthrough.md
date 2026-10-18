# How Figure 1b is computed, line by line

A complete trace of `figures/fig1b_regime.py`, from the parameter table to the
PNG. Every number below is real: it comes from running `figures/trace_fig1b.py`,
which reproduces the whole chain for a single draw and prints each intermediate.

Run it yourself:

```bash
python figures/trace_fig1b.py              # draw 0 at 20 mm/s
python figures/trace_fig1b.py --draw 42 --velocity 5
```

---

## 0 · The model in one sentence

> With ideal spoiling the only thing that survives from one TR to the next is
> the longitudinal magnetisation `Mz`, every event acting on it is an affine map
> `Mz → A·Mz + B`, so static tissue sits at that map's fixed point and blood that
> has received `n` excitations is that map applied `n` times — in closed form,
> with no loop over pulses.

Everything else is bookkeeping: which parameters, which geometry, how many
excitations each parcel of blood has seen.

## 1 · The files, and who calls whom

```
figures/fig1b_regime.py          the figure: draws, loops velocities, plots
├── tissues.py                   parameter intervals, the sampler, PD/T2* weighting
├── protocols.py                 TR and TE built from named timing components
├── skin.py                      Vessel, layer depths, laminar shell velocities
├── acquisition.py               Slab, path length through it, entry side
└── sequences/
    ├── base.py                  Sequence.echo, FlowGeometry, ReadoutTiming
    ├── prep.py                  saturation pulses, between_excitations
    └── tof.py                   TofSequence: static_signal, blood_signal
        └── affine.py            compose / power / fixed_point / apply
```

`affine.py` is the engine. It is 60 lines and contains no MR-specific code at all.

---

## 2 · Parameters are intervals (`tissues.py`)

A `Param` is an interval with provenance, not a number:

```python
@dataclass(frozen=True)
class Param:
    lo, hi, nominal, confidence, source, log
```

```
arterial_blood.t1_ms      1600 .. 1950   nominal 1650   medium
arterial_blood.t2star_ms    45 .. 75     nominal 65     medium
arterial_blood.pd_rel      1.1 .. 1.4    nominal 1.25   low
dermis.t2star_ms             5 .. 20     nominal 12     LOW
```

**`nominal` is never read by any figure.** It exists for debugging and for the
derivation notes. The figures use the whole interval, always.

`log=True` means the interval is sampled log-uniformly — right for a quantity
known only to within a factor (velocity, T2\*), wrong for one known to within an
offset.

### The draw

```python
params = collect_params(["arterial_blood", "hypodermis", "dermis"],
                        haemodynamic_keys=("target_vessel_velocity",
                                           "thermoregulatory_factor"))
# -> 20 parameters: 3 tissues x 6 fields, + 2 haemodynamic
unit = qmc.LatinHypercube(d=20, seed=20261014).random(96)   # 96 x 20 in [0,1)
```

then each column is mapped onto its own interval:

```python
def _scale(p, u):
    if p.log:
        return np.exp(np.log(p.lo) + u * (np.log(p.hi) - np.log(p.lo)))
    return p.lo + u * (p.hi - p.lo)
```

Draw 0 came out as:

| | T1 (ms) | T2\* (ms) | PD | fatsat_s |
|---|---|---|---|---|
| blood | 1830.5 | 54.03 | 1.297 | 1.000 |
| dermis | 1014.1 | 13.21 | 1.000 | 0.901 |
| fat | 428.8 | 40.07 | 1.598 | 0.212 |

**Why "jointly" matters.** One Latin hypercube covers all 20 parameters at once,
so draw 0 is a single coherent world — that blood sits beside *its own* dermis
and *its own* fat. This is what licenses the paired statistics the figure quotes
("brighter in 93 % of draws"). If each tissue were sampled separately, pairing
them would be meaningless.

**Why Latin hypercube and not plain random.** LHS stratifies each parameter into
`n` equal-probability bins and puts exactly one draw in each, so 96 draws cover
each interval evenly instead of clumping. `tests/test_tissues.py` asserts exactly
this property.

---

## 3 · TR and TE are built, not assumed (`protocols.py`)

Nothing in the model takes "TR = 35 ms" on faith. `fl3d1r_timing` derives each
gradient lobe from the voxel size at the scanner's gradient limits:

```python
def encoding_area(pixel_mm):            # gradient area to reach k-space edge
    return 1e6 / (2 * GAMMA_HZ_PER_MT * pixel_mm)

def min_lobe_ms(area, g):               # shortest trapezoid/triangle at the limits
    if area <= g.max_amplitude**2 / g.max_slew:
        return 2 * sqrt(area / g.max_slew)          # triangle
    return area / g.max_amplitude + g.max_amplitude / g.max_slew   # trapezoid
```

giving, for this figure:

```
RF 0.868 + rephaser 1.038 + prephaser 0.889 + readout 5.376 + spoiler 2.953
                                                       + dead time 23.875
                                                       = TR 35.000 ms
TE = RF/2 + rephaser + prephaser + readout/2         = 5.050 ms
```

Readout duration is `1000 / bandwidth_per_px_hz`. **68 % of this TR is dead
time** — that is the slack the readout-bandwidth finding exploits.

---

## 4 · The engine: affine maps (`affine.py`)

### Why an affine map

Under ideal spoiling, transverse magnetisation is destroyed before the next
pulse, so the entire state is the scalar `Mz`. Each event is linear in `Mz` plus
a constant:

| event | A | B |
|---|---|---|
| T1 relaxation over `t` | `exp(-t/T1)` | `M0(1 - exp(-t/T1))` |
| excitation at flip α | `cos α` | `0` |
| saturation leaving fraction `s` | `s` | `0` |

### Composition applies in order

```python
def walk(maps):
    a, b = 1.0, 0.0
    for a_i, b_i in maps:
        a, b = a_i * a, a_i * b + b_i
    return a, b
```

Verified by hand: with `m1=(0.5, 0.1)` then `m2=(0.8, 0.2)` on `Mz=1`:
`0.5·1+0.1 = 0.6`, then `0.8·0.6+0.2 = 0.68`. `compose(m1,m2)` gives
`(A=0.400, B=0.280)`, and `0.400·1 + 0.280 = 0.680`. ✓

### The three operations that do all the work

```python
def power(m, n):                        # n repetitions, closed form
    a_n = np.power(m[0], n)
    return a_n, m[1] * (1 - a_n) / (1 - m[0])

def fixed_point(m):                     # the steady state
    return m[1] / (1 - m[0])
```

`power` is the geometric series summed analytically. It needs no special cases:

| n | A^n | Mz from inflow 1.0 |
|---|---|---|
| 0 | 1.000000 | 1.000000 (identity) |
| 1 | 0.920246 | 0.940941 |
| 5 | 0.659964 | 0.748196 |
| **inf** | 0.000000 | **0.259480 = the fixed point exactly** |

That last row is the elegant part: **static tissue is just `n = ∞`.** There is no
separate code path for it. A vessel with velocity → 0 produces exactly
`static_signal`, which the trace verifies to 1e-6.

`A` and `B` may be NumPy arrays, so one map can carry the whole 96-draw ensemble.

### One TR, assembled

```python
def tr_map(self, tissue, flip_deg=None):
    relax = between_excitations(self.pulses(tissue), tissue, self.timing.tr_ms)
    return compose(excitation_map(flip_deg or self.flip_deg), relax)
```

For draw 0 with no saturation modules:

```
relaxation over TR   A = exp(-35/1830.5) = 0.98106,  B = 1 - A = 0.01894
excitation           A = cos(20°)        = 0.93969,  B = 0
composed one-TR map  A = 0.92190,                    B = 0.01894
fixed point B/(1-A)  = 0.24248      <- stagnant blood's Mz/M0
```

**Note the engine runs at unit M0.** `B = 1 - A`, not `PD·(1 - A)`. Proton
density is applied later, at the echo step. This keeps the maps tissue-agnostic.

### Saturation pulses and their order (`prep.py`)

A fat-sat or MTC pulse sits `delay_ms` *before* each excitation, so one TR is:

```
excite(α) → relax(TR - delay) → saturate(s) → relax(delay) → [next excitation]
```

`between_excitations` builds this by sorting pulses by delay and composing.
Verified against the hand-written composition for fat at 8 ms:
`compose(relax(27), saturate(0.259), relax(8))` matches to machine precision. ✓

The 8 ms tail matters: tissue recovers for 8 ms *after* being saturated, before
it is read out.

---

## 5 · Geometry (`skin.py`, `acquisition.py`)

```python
slab   = Slab(thickness_mm=12.0, orientation_deg=90.0)   # normal = [1, 0, 0]
vessel = Vessel(depth_mm=5.0, polar_deg=90.0, azimuth_deg=0.0, diameter_mm=0.3)
                                                         # direction = [1, 0, 0]
```

**The vessel direction is parallel to the slab normal**, so the vessel crosses
the slab end-on:

```python
path_length = slab.thickness_mm / |direction · normal| = 12.0 / 1.0 = 12.000 mm
```

This is the **shortest possible** crossing and puts the entire velocity into it,
so it is the best case for inflow. Any other angle lengthens the path and
saturates the blood further. The inversion the figure reports is therefore the
conservative end of the result.

(A vessel genuinely lying *in* the slab plane would be `azimuth_deg=90`,
`direction = [0,1,0]`, dot product 0 — it would never leave the slab at all.)

### Laminar flow

```python
def laminar_velocities(mean_velocity, n_shells):
    edges = np.linspace(0.0, 1.0, n_shells + 1)      # radius fraction
    inner, outer = edges[:-1], edges[1:]
    weights = outer**2 - inner**2                    # annular AREA
    velocities = 2 * mean_velocity * (1 - 0.5*(inner**2 + outer**2))
    return velocities, weights
```

`2·mean·(1 - r²)` is Poiseuille with `v_max = 2·v_mean`, area-averaged over each
annulus so the weighted mean is exact for any `n_shells`.

At a 20 mm/s mean:

| shell | v (mm/s) | area weight |
|---|---|---|
| 0 (centre) | 39.69 | 0.016 |
| 3 | 32.19 | 0.109 |
| 7 (wall) | **4.69** | **0.234** |

The wall shell is the slowest **and** carries the largest area weight: 23 % of
the lumen moves at under a quarter of the mean. This is precisely why plug-flow
models overestimate TOF signal, and it is a stated design decision in CLAUDE.md.

---

## 6 · Assembling the blood signal (`sequences/tof.py`)

### How many excitations

```python
def pulses_received(positions_mm, velocity_mm_s, tr_ms):
    step_mm = velocity_mm_s * tr_ms / 1000.0
    n = np.floor(positions_mm / step_mm)
    return np.where(step_mm > 0, n, np.inf)
```

At 20 mm/s one TR advances 0.70 mm. A spin 8.7 TRs into the slab is treated as
having had **8** pulses, never 8.7 — it is a step function per shell. With 8
shells × 32 positions the steps average out in the mean.

Velocity 0 → `inf` → the fixed point. This is the mechanism by which static
tissue needs no special case.

### The inflow boundary condition

```python
upstream = fixed_point(between_excitations(self.pulses(blood), blood, tr_ms))
```

Blood *upstream* of the slab is never excited (the excitation is slab-selective)
but **is** hit by any non-selective saturation pulse. With no modules, the fixed
point of relaxation alone is exactly **1.000** — fully relaxed.

> ### ⚠ The non-obvious consequence: MTC destroys the inflow
>
> | modules | blood's per-TR factor | upstream steady state |
> |---|---|---|
> | none | — | **1.000** |
> | fat saturation | `fatsat_s = 1.000` | **1.000** (blood is on-resonance, untouched) |
> | MTC | `mtc_s = 0.960` | **0.314** |
>
> A 4.0 % loss per TR against only 1.9 % T1 recovery per TR compounds to a
> steady state of 0.31. **Blood arrives at the slab already 69 % saturated,
> before a single excitation.** A non-selective MT pulse saturates blood
> everywhere in the body, including upstream, so it trades background
> suppression against the very inflow TOF depends on.
>
> This is a model prediction, not a measurement — but it is consistent with the
> 15 October result that MTC tied on CNR per √time, and it is a better
> explanation of that null than "dermis has a small bound pool".

### The main loop that isn't a loop

```python
shell_factors, weights = laminar_velocities(1.0, flow.n_shells)
velocities = shell_factors[:, None] * np.atleast_1d(flow.mean_velocity_mm_s)
n          = pulses_received(positions[:, None, None], velocities[None], tr_ms)
inflow     = self.inflow_mz(blood, flow, velocities)
mz         = apply(power(self.tr_map(blood), n), inflow[None])
out        = np.einsum("psn,s->pn", self.echo(mz, flip, blood), weights)
```

Shapes, verified:

```
positions                (32,)       P = positions along the vessel
shell_factors            (8,)        S = laminar shells
velocities  (S,N)        (8, 96)     N = parameter draws
positions[:,None,None]   (32, 1, 1)
n = (P,1,1) vs (1,S,N) → (32, 8, 96)   broadcast, no Python loop
Mz                       (32, 8, 96)
echo                     (32, 8, 96)
einsum("psn,s->pn")    → (32, 96)       area-weighted over shells
```

One call evaluates 24 576 spin histories. The only Python loop in the whole
engine is over *pulses*, and only when TONE is on (which this figure does not
use, `tone_pct = 0`).

### Mz → echo signal

```python
def echo(self, mz_over_m0, flip_deg, tissue):
    return mz_over_m0 * np.sin(np.radians(flip_deg)) * signal_weighting(tissue, self.timing.te_ms)

def signal_weighting(tissue, te_ms):
    return tissue.pd_rel * np.exp(-te_ms / tissue.t2star_ms)
```

For draw 0 at the slab exit, centreline:

```
weighting = 1.2973 · exp(-5.050 / 54.03)           = 1.18152
echo      = 0.63771 · sin(20°) · 1.18152           = 0.25770
```

**This is where the y-axis unit is set.** Dermis has `pd_rel = 1.0` *by
definition* (it is the reference tissue), so "1" on the axis means the signal a
fully relaxed dermis would give with no T2\* decay.

### Along the path, and the averaging choice

`blood_signal` returns one value per position. Along the 12 mm crossing for
draw 0 at 20 mm/s: **0.404 at entry → 0.173 at exit.** The figure plots the
**mean, 0.25192**.

> That mean mixes fresh blood at the entry face with saturated blood at the
> exit. A maximum intensity projection would weight the bright entry end more
> heavily, so the figure is pessimistic relative to how a MIP would look.

---

## 7 · From signal to plot (`fig1b_regime.py`)

```python
blood = np.array([[blood_at(plain, s["arterial_blood"], v) for s in samples]
                  for v in VELOCITIES])          # (64 velocities, 96 draws)
lo, mid, hi = np.percentile(blood, [5, 50, 95], axis=-1)
```

The orange line is `mid`; the band is `lo..hi`. At 20 mm/s: median 0.2454,
5 % 0.2178, 95 % 0.2731.

The four flat references come from `static_signal` on each tissue, each with its
own 5–95 % band (`reference(..., band=True)` is now the default — see the commit
"Show uncertainty on every level").

### The inversion statistics

Two different numbers, and they answer different questions:

```python
xs   = crossing_velocities(...)   # per draw, bisection in log velocity
frac = mean(blood_at(s, v_s) < static(s))   # each draw at ITS OWN velocity
```

- **`xs`**: for each draw, *at what velocity* would it invert → 16 mm/s [5–46].
  Found by `scipy.optimize.brentq` on log velocity, **not** read off the
  plotting grid — otherwise the quoted bounds move when the grid is refined
  (this is what shifted an earlier "7–44" to "5–46").
- **`frac`**: pairing each draw's tissue parameters with *that same draw's*
  velocity, how many are below the threshold → 57 %.

---

## 8 · What the figure does not model

Each of these is a real limitation. The direction of the error is given where it
is known.

| Assumption | Where | Direction of error |
|---|---|---|
| **Ideal rectangular slab profile** | `tof.py` docstring | Real RF gives reduced flip at the slab edges, so real blood is *less* saturated near the faces. Figure is pessimistic. |
| **Ideal spoiling** | the whole affine treatment | Residual transverse magnetisation would add coherent pathways not modelled at all. |
| **`n = floor(...)` quantisation** | `pulses_received` | Unbiased in the mean; a step function per shell. |
| **Slab-mean along the path** | `blood_at` | Mixes fresh and saturated blood. A MIP weights the entry end more, so the figure is pessimistic. |
| **End-on crossing** | the chosen geometry | The best case for inflow. Any other angle deepens the inversion. Figure is conservative. |
| **Velocity ⊥ diameter** | the LHS samples them independently although v ∝ d | Breaks a real correlation; understates how bad small vessels are, since they are both slower and more partial-volumed. Conservative. |
| **MTC as a per-pulse scalar** | `prep.py` docstring | Real MT builds up in the bound pool and transfers continuously. `mtc_s` is an effective number, not a pulse property. |
| **No noise, no voxel** | — | This figure is *signal*, not CNR. A 6× ratio over near-zero suppressed fat says nothing about clearing the noise floor. Detectability is a separate calculation (`detectability.py`, `noise.py`). |

---

## 8b · Is the laminar flow model physiologically right?

### The fluid mechanics is not in doubt

Three dimensionless numbers decide whether Poiseuille applies, and all three
clear their thresholds by orders of magnitude:

| d (µm) | v (mm/s) | Reynolds | entrance length | Womersley α |
|---|---|---|---|---|
| 100 | 5 | 0.15 | 0.9 µm | 0.076 |
| 300 | 15 | 1.4 | 25 µm | 0.227 |
| 500 | 24 | 3.6 | 109 µm | 0.378 |
| 1930 (perforator) | 89 | 52 | 6.0 mm | 1.458 |

- **Turbulence needs Re > ~2000.** We are at Re ~1. Not close.
- **Entrance length** is how far the parabola needs to re-develop after a
  branch or bend. At 25 µm it is negligible against a 12 mm path, so the
  profile is fully developed essentially everywhere.
- **Womersley α < 3** means quasi-steady: the profile is parabolic at every
  instant of the cardiac cycle, just scaled up and down.

### The two real departures, and their measured size

**Blunting from the cell-free layer.** Red cells migrate centrally, leaving a
1–5 µm plasma layer at the wall ([Fedosov 2010](https://doi.org/10.1111/j.1549-8719.2010.00056.x)),
which blunts the profile. The effect is significant below ~300 µm
([Stergiou 2019](https://doi.org/10.3390/fluids4030143)) and at 100–500 µm the
layer is only a few percent of the diameter. Tested on the real pipeline with
`v = v_max(1 - (r/R)^k)`, all normalised to the same mean velocity:

| k | profile | wall shell / mean | blood @ 15 mm/s | vs Poiseuille |
|---|---|---|---|---|
| 2 | Poiseuille | 0.234 | 0.2226 | 1.00× |
| 4 | blunted | 0.324 | 0.2276 | 1.02× |
| 9 | very blunt | 0.492 | 0.2318 | 1.04× |
| ∞ | **plug flow** | 1.000 | 0.2357 | **1.06×** |

Even the most extreme possible case — full plug flow — moves the slab-mean
signal by 6 % and the inversion velocity from 16.5 to 14.2 mm/s. Realistic
blunting moves it by 2 %.

**Pulsatility.** A multi-minute scan samples every cardiac phase, so the image
signal is the *time average* of the signal, not the signal at the mean velocity.
The signal-versus-velocity curve is concave, so by Jensen's inequality those
differ. With RI = 0.71 from the perforator data:

| mean velocity | signal at mean | time-averaged | difference |
|---|---|---|---|
| 5 mm/s | 0.1543 | 0.1519 | −1.5 % |
| 15 mm/s | 0.2226 | 0.2137 | −4.0 % |
| 40 mm/s | 0.2921 | 0.2815 | −3.6 % |

A few percent, and in the *opposite* direction to blunting, so the two largely
cancel.

### What actually drives the result is the geometry, not the profile

The same test on vessel angle, which sets the path length through the slab:

| azimuth | path (mm) | blood @ 15 mm/s | vs end-on | inversion velocity |
|---|---|---|---|---|
| 0° (end-on) | 12.0 | 0.2226 | 1.00× | **16.5 mm/s** |
| 45° | 17.0 | 0.1991 | 0.89× | 23.3 mm/s |
| 70° | 35.1 | 0.1553 | 0.70× | **48.2 mm/s** |

**The inversion velocity nearly triples with angle, while it barely moves with
profile shape.** At 70° the inversion sits above the entire target velocity
range, so every target vessel would be darker than unsuppressed fat.

So the assumption to interrogate is not Poiseuille. It is that a vessel is a
**single straight constant-calibre segment at one angle**. Real cutaneous
vessels are tortuous and branch — in the dermal plexus, at intervals well under
the 12 mm slab. Both tortuosity and branching lengthen the effective path and
slow the blood, which deepens the inversion. The figure's end-on geometry is
the most favourable case.

### One caveat on what this does and does not license

CLAUDE.md (Decision 6) argues laminar flow is "a real finding, not a
refinement" because plug flow overestimates TOF signal and makes vessels appear
narrower. The 6 % above tests only the first half, for the **slab-mean signal**
on this figure. The apparent-width claim is a different quantity — near-wall
blood being dark makes the vessel's bright core narrower than the lumen — and it
is not tested here. Laminar shells may well matter more for `detectability.py`
than they do for Figure 1b. Do not carry the 6 % across.

## 9 · How to check it yourself

```bash
python figures/trace_fig1b.py --draw 7 --velocity 12   # every intermediate
python -m pytest tests/ -q                             # 125 tests
python figures/fig1b_regime.py                         # regenerate, prints all stats
```

Three checks worth doing by hand:

1. **The fixed point.** Take any draw's blood T1, compute
   `A = exp(-TR/T1)·cos(α)`, `B = 1 - exp(-TR/T1)`, and verify `B/(1-A)` equals
   the stagnant line.
2. **The static limit.** Set velocity to 1e-9 and confirm `blood_signal`
   collapses onto `static_signal`.
3. **The laminar weights.** Confirm `sum(weights) == 1` and
   `sum(velocities·weights) == mean_velocity`.

No measured quantity enters any of this. Acquisition settings (TR, TE, flip,
bandwidth, voxel, slab) are inputs; measured signal, SNR, contrast and calibre
are not. See the hard constraint in `CLAUDE.md`.
