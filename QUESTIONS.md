# Questions

This toolbox exists to answer the questions below. The list is the stopping criterion: the toolbox is finished when it can answer these, not when it feels complete.

All of them concern the Siemens `fl3d1r` TOF product sequence on a 3 T Prisma with a 4-channel surface coil, in the sensitivity-limited regime: with fat saturation on, the background at 1–10 mm depth sits near the noise floor, so absolute CNR, not a contrast ratio, decides the protocol.

1. What background SNR is needed to detect a vessel of diameter *d* at voxel size *a*, and how does that trade against scan time?
2. At what voxel size does a vessel of diameter *d* stop being partial-volumed, and when does its apparent width start reporting its true width rather than the point-spread function?
3. What flip angle maximises absolute vessel CNR per √(scan time) at the achievable TR, and how much does that answer move across the tissue parameter intervals?
4. How much does fat saturation change blood-to-background CNR at 1–10 mm depth, and how much of the apparent gain is suppression of fat rather than enhancement of blood?
5. What does the TONE ramp do to the through-slab signal profile of static tissue and of blood at a given velocity, and is that profile distinguishable from the inflow fade?
6. What difference should the travelling saturation band make between its two positions, as a function of velocity and vessel orientation?
7. Which tissue parameter dominates the uncertainty in the answers above, and therefore what should be measured first?

## Notes

- **T2-prep is not on this list, deliberately.** It is retained in the codebase as a comparison calculation. The toolbox does not have to decide whether T2-prep wins; it has to be able to state what T2-prep would give at these tissue parameters and what it would cost in scan time, when asked.
- **Every answer above is a prediction from literature parameters.** None of it is fitted to the in vivo data of 8 and 13 October 2026. Comparison with those measurements is a separate, later step whose predictions are committed before the comparison is made.
