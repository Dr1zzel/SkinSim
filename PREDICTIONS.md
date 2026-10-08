# Predictions

Generated 2026-10-08T12:14:44+02:00 from commit 10e4615 (with uncommitted changes) by `predictions.py`.

Blood/background signal ratio, arterial blood, all tissue values nominal. Each cell gives the ratio at slab entry / averaged over 100 points along the slab / at slab exit. 'static' is the v -> 0 limit.

- Readout: TR 40.34 ms, TE 5.95 ms, flip 18 deg (PD 5 deg); dead time 28.89 ms is an unmodelled remainder.
- Geometry: slab 11.52 mm, transversal (orientation 90 deg); vessel along the limb (polar 90, azimuth 0 deg), path 11.52 mm; depths (1.0, 5.0) mm; 8 laminar shells; max_path_mm 30 (unjustified, inactive here).
- Segmented protocols: 64 lines, recovery 0 ms, centric; preparation non-selective, so inflowing blood arrives prepared but unsaturated.
- IR fat null: 200.7 ms segmented vs 263.4 ms for T1 ln 2 (-62.7 ms).
- T2-prep uses one blood T2 for every refocusing interval; Zhao (MRM 2007) shows it varies, so these predictions do not yet carry that dependence.

## Tissue values used

| tissue | T1 ms | T2 ms | T2* ms | PD |
|---|---|---|---|---|
| arterial blood | 1650 (MEDIUM) | 180 (LOW) | 65 (MEDIUM) | 1.25 (LOW) |
| dermis | 1200 (VERY_LOW) | 45 (VERY_LOW) | 15 (VERY_LOW) | 1 (HIGH) |
| hypodermis (subcutaneous fat) | 380 (MEDIUM) | 70 (LOW) | 25 (VERY_LOW) | 1.4 (LOW) |

## Predicted signals

Echo signal in units of fully relaxed dermis magnetisation (proton density 1). Blood does not depend on depth; 'blood mean' averages along the slab. Compare these, not ratios, wherever a background is nulled: a measured ratio against a nulled background is set by noise and imperfect nulling.

| protocol | dermis | fat | blood static | blood entry | blood mean, 1 mm/s | blood mean, 2.5 mm/s | blood mean, 7.5 mm/s | blood mean, 25 mm/s |
|---|---|---|---|---|---|---|---|---|
| 1. TOF | 0.0855 | 0.2373 | 0.1184 | 0.3525 | 0.1308 | 0.1476 | 0.1937 | 0.2637 |
| 2. PD | 0.0527 | 0.0930 | 0.0862 | 0.0994 | 0.0879 | 0.0900 | 0.0935 | 0.0968 |
| 3. IR, TI 200.7 ms (nulls fat) | 0.0398 | 0.0000 | 0.0631 | 0.1742 | 0.0691 | 0.0774 | 0.1041 | 0.1468 |
| 3. IR, TI 150 ms | 0.0504 | 0.0487 | 0.0760 | 0.1880 | 0.0815 | 0.0889 | 0.1136 | 0.1557 |
| 3. IR, TI 200 ms | 0.0399 | 0.0006 | 0.0633 | 0.1744 | 0.0693 | 0.0775 | 0.1042 | 0.1469 |
| 3. IR, TI 250 ms | 0.0299 | 0.0415 | 0.0510 | 0.1611 | 0.0574 | 0.0664 | 0.0950 | 0.1371 |
| 3. IR, TI 300 ms | 0.0202 | 0.0784 | 0.0390 | 0.1480 | 0.0459 | 0.0556 | 0.0859 | 0.1270 |
| 4. T2-prep, 40 ms (10 ms refocusing interval) | 0.0358 | 0.1367 | 0.0954 | 0.2684 | 0.1050 | 0.1180 | 0.1543 | 0.2097 |
| 4. T2-prep, 60 ms (15 ms refocusing interval) | 0.0232 | 0.1038 | 0.0854 | 0.2354 | 0.0940 | 0.1055 | 0.1377 | 0.1870 |
| 4. T2-prep, 80 ms (20 ms refocusing interval) | 0.0152 | 0.0791 | 0.0765 | 0.2069 | 0.0840 | 0.0943 | 0.1231 | 0.1670 |
| 4. T2-prep, 100 ms (25 ms refocusing interval) | 0.0100 | 0.0606 | 0.0685 | 0.1823 | 0.0753 | 0.0844 | 0.1102 | 0.1493 |

## Predicted blood/background ratio

### 1. TOF

Time per line 40.34 ms.

| mean v (mm/s) | dermis: entry / mean / exit | hypodermis (subcutaneous fat): entry / mean / exit |
|---|---|---|
| 1 | 4.124 / 1.531 / 1.385 | 1.485 / 0.551 / 0.499 |
| 2.5 | 4.124 / 1.727 / 1.392 | 1.485 / 0.622 / 0.501 |
| 7.5 | 4.124 / 2.266 / 1.613 | 1.485 / 0.816 / 0.581 |
| 25 | 4.124 / 3.086 / 2.427 | 1.485 / 1.111 / 0.874 |
| 0 (static) | 1.385 / 1.385 / 1.385 | 0.499 / 0.499 / 0.499 |

### 2. PD

Time per line 40.34 ms.

| mean v (mm/s) | dermis: entry / mean / exit | hypodermis (subcutaneous fat): entry / mean / exit |
|---|---|---|
| 1 | 1.885 / 1.666 / 1.634 | 1.069 / 0.945 / 0.927 |
| 2.5 | 1.885 / 1.705 / 1.649 | 1.069 / 0.967 / 0.935 |
| 7.5 | 1.885 / 1.772 / 1.711 | 1.069 / 1.005 / 0.970 |
| 25 | 1.885 / 1.835 / 1.794 | 1.069 / 1.040 / 1.017 |
| 0 (static) | 1.634 / 1.634 / 1.634 | 0.926 / 0.926 / 0.926 |

### 3. IR, TI 200.7 ms (nulls fat)

Time per line 43.48 ms.

| mean v (mm/s) | dermis: entry / mean / exit | hypodermis (subcutaneous fat): entry / mean / exit |
|---|---|---|
| 1 | 4.377 / 1.736 / 1.585 | 6.05e+08 / 2.40e+08 / 2.19e+08 |
| 2.5 | 4.377 / 1.944 / 1.577 | 6.05e+08 / 2.69e+08 / 2.18e+08 |
| 7.5 | 4.377 / 2.614 / 1.735 | 6.05e+08 / 3.61e+08 / 2.40e+08 |
| 25 | 4.377 / 3.688 / 2.955 | 6.05e+08 / 5.10e+08 / 4.08e+08 |
| 0 (static) | 1.585 / 1.585 / 1.585 | 2.19e+08 / 2.19e+08 / 2.19e+08 |

### 3. IR, TI 150 ms

Time per line 42.68 ms.

| mean v (mm/s) | dermis: entry / mean / exit | hypodermis (subcutaneous fat): entry / mean / exit |
|---|---|---|
| 1 | 3.727 / 1.615 / 1.506 | 3.862 / 1.674 / 1.561 |
| 2.5 | 3.727 / 1.763 / 1.500 | 3.862 / 1.826 / 1.554 |
| 7.5 | 3.727 / 2.252 / 1.600 | 3.862 / 2.333 / 1.658 |
| 25 | 3.727 / 3.086 / 2.452 | 3.862 / 3.198 / 2.541 |
| 0 (static) | 1.506 / 1.506 / 1.506 | 1.560 / 1.560 / 1.560 |

### 3. IR, TI 200 ms

Time per line 43.47 ms.

| mean v (mm/s) | dermis: entry / mean / exit | hypodermis (subcutaneous fat): entry / mean / exit |
|---|---|---|
| 1 | 4.366 / 1.734 / 1.584 | 272.607 / 108.254 / 98.910 |
| 2.5 | 4.366 / 1.941 / 1.576 | 272.607 / 121.222 / 98.408 |
| 7.5 | 4.366 / 2.609 / 1.733 | 272.607 / 162.892 / 108.225 |
| 25 | 4.366 / 3.678 / 2.949 | 272.607 / 229.644 / 184.156 |
| 0 (static) | 1.584 / 1.584 / 1.584 | 98.909 / 98.909 / 98.909 |

### 3. IR, TI 250 ms

Time per line 44.25 ms.

| mean v (mm/s) | dermis: entry / mean / exit | hypodermis (subcutaneous fat): entry / mean / exit |
|---|---|---|
| 1 | 5.392 / 1.922 / 1.706 | 3.883 / 1.384 / 1.228 |
| 2.5 | 5.392 / 2.223 / 1.695 | 3.883 / 1.601 / 1.221 |
| 7.5 | 5.392 / 3.180 / 1.960 | 3.883 / 2.290 / 1.411 |
| 25 | 5.392 / 4.591 / 3.754 | 3.883 / 3.306 / 2.704 |
| 0 (static) | 1.706 / 1.706 / 1.706 | 1.228 / 1.228 / 1.228 |

### 3. IR, TI 300 ms

Time per line 45.03 ms.

| mean v (mm/s) | dermis: entry / mean / exit | hypodermis (subcutaneous fat): entry / mean / exit |
|---|---|---|
| 1 | 7.325 / 2.272 / 1.930 | 1.888 / 0.586 / 0.497 |
| 2.5 | 7.325 / 2.753 / 1.913 | 1.888 / 0.710 / 0.493 |
| 7.5 | 7.325 / 4.251 / 2.374 | 1.888 / 1.096 / 0.612 |
| 25 | 7.325 / 6.284 / 5.180 | 1.888 / 1.620 / 1.335 |
| 0 (static) | 1.930 / 1.930 / 1.930 | 0.497 / 0.497 / 0.497 |

### 4. T2-prep, 40 ms (10 ms refocusing interval)

Time per line 41.04 ms.

| mean v (mm/s) | dermis: entry / mean / exit | hypodermis (subcutaneous fat): entry / mean / exit |
|---|---|---|
| 1 | 7.506 / 2.936 / 2.668 | 1.963 / 0.768 / 0.698 |
| 2.5 | 7.506 / 3.301 / 2.679 | 1.963 / 0.863 / 0.701 |
| 7.5 | 7.506 / 4.314 / 3.081 | 1.963 / 1.129 / 0.806 |
| 25 | 7.506 / 5.865 / 4.662 | 1.963 / 1.534 / 1.220 |
| 0 (static) | 2.668 / 2.668 / 2.668 | 0.698 / 0.698 / 0.698 |

### 4. T2-prep, 60 ms (15 ms refocusing interval)

Time per line 41.36 ms.

| mean v (mm/s) | dermis: entry / mean / exit | hypodermis (subcutaneous fat): entry / mean / exit |
|---|---|---|
| 1 | 10.135 / 4.045 / 3.679 | 2.267 / 0.905 / 0.823 |
| 2.5 | 10.135 / 4.541 / 3.692 | 2.267 / 1.016 / 0.826 |
| 7.5 | 10.135 / 5.931 / 4.230 | 2.267 / 1.326 / 0.946 |
| 25 | 10.135 / 8.052 / 6.388 | 2.267 / 1.801 / 1.429 |
| 0 (static) | 3.679 / 3.679 / 3.679 | 0.823 / 0.823 / 0.823 |

### 4. T2-prep, 80 ms (20 ms refocusing interval)

Time per line 41.67 ms.

| mean v (mm/s) | dermis: entry / mean / exit | hypodermis (subcutaneous fat): entry / mean / exit |
|---|---|---|
| 1 | 13.619 / 5.530 / 5.036 | 2.615 / 1.062 / 0.967 |
| 2.5 | 13.619 / 6.205 / 5.052 | 2.615 / 1.191 / 0.970 |
| 7.5 | 13.619 / 8.101 / 5.785 | 2.615 / 1.555 / 1.111 |
| 25 | 13.619 / 10.994 / 8.803 | 2.615 / 2.111 / 1.690 |
| 0 (static) | 5.036 / 5.036 / 5.036 | 0.967 / 0.967 / 0.967 |

### 4. T2-prep, 100 ms (25 ms refocusing interval)

Time per line 41.98 ms.

| mean v (mm/s) | dermis: entry / mean / exit | hypodermis (subcutaneous fat): entry / mean / exit |
|---|---|---|
| 1 | 18.143 / 7.489 / 6.820 | 3.010 / 1.242 / 1.131 |
| 2.5 | 18.143 / 8.395 / 6.841 | 3.010 / 1.393 / 1.135 |
| 7.5 | 18.143 / 10.965 / 7.815 | 3.010 / 1.819 / 1.296 |
| 25 | 18.143 / 14.858 / 11.895 | 3.010 / 2.465 / 1.973 |
| 0 (static) | 6.820 / 6.820 / 6.820 | 1.131 / 1.131 / 1.131 |
