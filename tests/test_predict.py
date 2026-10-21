"""The composition layer: every protocol runs end to end, and the joins hold.

The point of these tests is reachability as much as correctness. Each half of
the toolbox was already unit-tested in isolation; what went untested was that
the halves compose, which is why the efficiency metric could sit unused.
"""
from __future__ import annotations


import numpy as np
import pytest

from detect import ROSE, blood_fraction, cnr_efficiency
from geometry import Vessel
from predict import Prediction, cross_section_voxel, predict
from protocols import ALL_PROTOCOLS, MEASURED
from sequence import FlowGeometry
from tissues import ACQUISITION, TISSUES, TissueSample, sample_tissues

SEED, N_DRAWS = 20261014, 32
DEPTH_MM, DIAMETER_MM = 5.0, 0.3  # inside the hypodermis, mid-target calibre
KEYS = ["arterial_blood", "dermis", "hypodermis"]


@pytest.fixture(scope="module")
def drawn():
    """Return one joint Latin-hypercube draw of the tissues, velocity and background structure."""
    tissues, others = sample_tissues(
        KEYS, N_DRAWS, SEED, haemodynamic_keys=("target_vessel_velocity", "thermoregulatory_factor"),
        acquisition_keys=("background_structure_rel",))
    velocity = others["target_vessel_velocity"] * others["thermoregulatory_factor"]
    return tissues, velocity, others["background_structure_rel"]


def flow_for(protocol, velocity, polar_deg=90.0, azimuth_deg=60.0):
    """Return a target-calibre vessel in the hypodermis crossing the protocol's slab."""
    vessel = Vessel(depth_mm=DEPTH_MM, polar_deg=polar_deg, azimuth_deg=azimuth_deg,
                    diameter_mm=DIAMETER_MM)
    return FlowGeometry(vessel, protocol.slab(), velocity, 8, 30.0, entry_z_frac=0.5)


def prediction_for(protocol, drawn, **kwargs):
    """Return the Prediction for a protocol over the drawn ensemble, sampled along the whole path."""
    tissues, velocity, rel = drawn
    flow = flow_for(protocol, velocity)
    positions = np.linspace(0.0, flow.path_mm(), 12)
    return predict(protocol, tissues["arterial_blood"], tissues["hypodermis"], flow, positions,
                   background_structure_rel=kwargs.pop("background_structure_rel", rel), **kwargs)


# --- every protocol composes ---------------------------------------------------


@pytest.mark.parametrize("name", list(ALL_PROTOCOLS), ids=list(ALL_PROTOCOLS))
def test_every_protocol_predicts_end_to_end(name, drawn):
    """Nothing in ALL_PROTOCOLS fails to produce finite signals and a finite efficiency."""
    pred = prediction_for(ALL_PROTOCOLS[name], drawn)
    assert pred.blood.shape == (12, N_DRAWS)
    assert np.all(np.isfinite(pred.blood)) and np.all(pred.blood > 0)
    assert np.all(np.isfinite(pred.background)) and np.all(pred.background > 0)
    assert np.all(np.isfinite(pred.cnr_efficiency))
    assert 0 < pred.blood_fraction <= 1


@pytest.mark.parametrize("name", list(ALL_PROTOCOLS), ids=list(ALL_PROTOCOLS))
def test_tone_protocols_broadcast_the_background_along_the_vessel(name, drawn):
    """Under TONE the background carries a position axis; without it one number serves every position."""
    pred = prediction_for(ALL_PROTOCOLS[name], drawn)
    expected = (12, N_DRAWS) if ALL_PROTOCOLS[name].tone_pct else (N_DRAWS,)
    assert pred.background.shape == expected
    assert pred.contrast.shape == (12, N_DRAWS)


def test_tone_makes_the_background_vary_with_position(drawn):
    """The flip ramp changes the background along the slab, so a single background number would be wrong."""
    with_tone = prediction_for(MEASURED["13oct_0.35mm"], drawn)
    without = prediction_for(ALL_PROTOCOLS["tone_off"], drawn)
    assert np.ptp(np.median(with_tone.background, axis=-1)) > 0
    assert np.ndim(without.background) == 1


# --- the joins ----------------------------------------------------------------


def test_cnr_efficiency_is_the_contrast_per_sqrt_time_per_line(drawn):
    """Prediction.cnr_efficiency is detect.cnr_efficiency of its own signals, not a reimplementation."""
    pred = prediction_for(MEASURED["13oct_0.35mm"], drawn)
    expected = cnr_efficiency(pred.blood, pred.background, pred.time_per_line_ms)
    assert pred.cnr_efficiency == pytest.approx(expected, rel=1e-12)
    hand = (pred.blood - pred.background) / np.sqrt(pred.time_per_line_ms / 1000.0)
    assert pred.cnr_efficiency == pytest.approx(hand, rel=1e-12)


def test_structure_fraction_is_derived_from_the_background(drawn):
    """The absolute texture is the relative structure times the simulated background, never a stored constant."""
    pred = prediction_for(MEASURED["13oct_0.35mm"], drawn, background_structure_rel=0.08)
    assert pred.structure_fraction == pytest.approx(0.08 * pred.background, rel=1e-12)


def test_fat_saturation_suppresses_background_and_its_texture_together(drawn):
    """Fat saturation lowers the texture floor by the same factor as the background, which is why it decides detection."""
    on = prediction_for(MEASURED["13oct_0.35mm"], drawn, background_structure_rel=0.08)
    off = prediction_for(ALL_PROTOCOLS["fatsat_off"], drawn, background_structure_rel=0.08)
    bkg_ratio = np.median(off.background) / np.median(on.background)
    tex_ratio = np.median(off.structure_fraction) / np.median(on.structure_fraction)
    assert bkg_ratio > 5
    assert tex_ratio == pytest.approx(bkg_ratio, rel=1e-9)


def test_unsaturated_fat_inverts_the_contrast(drawn):
    """Without fat saturation the hypodermis is brighter than blood, so the vessel is dark, not bright."""
    off = prediction_for(ALL_PROTOCOLS["fatsat_off"], drawn)
    on = prediction_for(MEASURED["13oct_0.35mm"], drawn)
    assert np.median(off.contrast) < 0 < np.median(on.contrast)


def test_texture_caps_cnr_however_high_the_snr(drawn):
    """CNR tends to f |contrast| / texture as reference SNR grows, instead of growing without bound."""
    pred = prediction_for(MEASURED["13oct_0.35mm"], drawn, background_structure_rel=0.08)
    ceiling = pred.blood_fraction * np.abs(pred.contrast) / pred.structure_fraction
    assert np.all(pred.cnr(1e9) <= ceiling * (1 + 1e-6))
    assert pred.cnr(1e9) == pytest.approx(ceiling, rel=1e-6)
    assert np.all(pred.cnr(1e4) < pred.cnr(1e9))


def test_snr_required_is_exactly_the_snr_that_reaches_the_threshold(drawn):
    """Feeding snr_required back into cnr returns the Rose threshold, for every draw that has a finite answer."""
    pred = prediction_for(MEASURED["13oct_0.35mm"], drawn, background_structure_rel=0.08)
    need = pred.snr_required
    finite = np.isfinite(need)
    assert finite.any()
    assert pred.cnr(need)[finite] == pytest.approx(ROSE, rel=1e-9)


def test_detected_agrees_with_the_threshold_on_cnr(drawn):
    """detected() is the Rose comparison on cnr(), with no second opinion about the threshold."""
    pred = prediction_for(MEASURED["13oct_0.35mm"], drawn, background_structure_rel=0.08)
    assert np.array_equal(pred.detected(50.0), pred.cnr(50.0) >= ROSE)
    assert np.all(pred.detected(1e6, cnr_threshold=0.0))
    assert not np.any(pred.detected(0.0))


def test_zero_texture_leaves_cnr_proportional_to_snr(drawn):
    """With no background structure the peak-voxel CNR is simply f |contrast| SNR."""
    pred = prediction_for(MEASURED["13oct_0.35mm"], drawn, background_structure_rel=0.0)
    expected = pred.blood_fraction * np.abs(pred.contrast) * 25.0
    assert pred.cnr(25.0) == pytest.approx(expected, rel=1e-12)


# --- the voxel the vessel actually sees ---------------------------------------


def test_cross_section_voxel_is_exact_when_the_vessel_crosses_the_slab_end_on():
    """A vessel along the slab normal sees the two in-plane extents."""
    protocol = MEASURED["13oct_0.35mm"]  # slab normal along the leg, orientation 90 deg
    vessel = Vessel(depth_mm=DEPTH_MM, polar_deg=90.0, azimuth_deg=0.0, diameter_mm=DIAMETER_MM)
    assert cross_section_voxel(protocol, vessel) == pytest.approx(
        (protocol.inplane_mm, protocol.inplane_mm))


def test_cross_section_voxel_is_exact_when_the_vessel_lies_in_the_slab_plane():
    """A vessel perpendicular to the slab normal sees one in-plane extent and the partition thickness."""
    protocol = MEASURED["13oct_0.35mm"]
    vessel = Vessel(depth_mm=DEPTH_MM, polar_deg=90.0, azimuth_deg=90.0, diameter_mm=DIAMETER_MM)
    assert cross_section_voxel(protocol, vessel) == pytest.approx(
        (protocol.inplane_mm, protocol.partition_mm))


@pytest.mark.parametrize("azimuth_deg", [0.0, 20.0, 45.0, 70.0, 90.0])
def test_cross_section_voxel_area_is_the_projected_shadow_of_the_voxel(azimuth_deg):
    """The product of the returned extents is the voxel box's shadow area projected along the vessel."""
    protocol = MEASURED["13oct_0.35mm"]
    vessel = Vessel(depth_mm=DEPTH_MM, polar_deg=90.0, azimuth_deg=azimuth_deg,
                    diameter_mm=DIAMETER_MM)
    cos = abs(float(np.dot(vessel.direction(), protocol.slab().normal())))
    sin = np.sqrt(1.0 - cos**2)
    a, p = protocol.inplane_mm, protocol.partition_mm
    shadow = a * p * sin + a * a * cos
    ay, az = cross_section_voxel(protocol, vessel)
    assert ay * az == pytest.approx(shadow, rel=1e-12)


def test_an_oblique_vessel_sees_a_larger_voxel_than_either_limit():
    """A box's shadow is widest along a diagonal, so the oblique case is not bracketed by the two limits."""
    protocol = MEASURED["13oct_0.35mm"]
    area = {}
    for azimuth_deg in (0.0, 45.0, 90.0):
        vessel = Vessel(depth_mm=DEPTH_MM, polar_deg=90.0, azimuth_deg=azimuth_deg,
                        diameter_mm=DIAMETER_MM)
        ay, az = cross_section_voxel(protocol, vessel)
        area[azimuth_deg] = ay * az
    assert area[45.0] > max(area[0.0], area[90.0])


def test_a_finer_voxel_raises_the_blood_fraction(drawn):
    """Shrinking the voxel towards the vessel concentrates it, which is the gain a finer protocol buys."""
    coarse = prediction_for(MEASURED["13oct_0.46mm"], drawn)
    fine = prediction_for(ALL_PROTOCOLS["inplane_0.18mm"], drawn)
    assert fine.blood_fraction > coarse.blood_fraction
    assert blood_fraction(DIAMETER_MM, fine.voxel_mm) == pytest.approx(fine.blood_fraction)


# --- guards and layering ------------------------------------------------------


def test_a_background_off_the_vessel_depth_is_refused(drawn):
    """The background must be the tissue the vessel's depth puts it in, as the sequence layer also insists."""
    tissues, velocity, rel = drawn
    protocol = MEASURED["13oct_0.35mm"]
    flow = flow_for(protocol, velocity)
    with pytest.raises(ValueError, match="hypodermis"):
        predict(protocol, tissues["arterial_blood"], tissues["dermis"], flow, np.array([0.0]),
                background_structure_rel=rel)


def test_predict_accepts_a_single_nominal_sample():
    """A scalar sample needs no sample axis, so the composition works for a quick sanity check too."""
    protocol = MEASURED["13oct_0.35mm"]
    blood = TissueSample.nominal(TISSUES["arterial_blood"])
    background = TissueSample.nominal(TISSUES["hypodermis"])
    flow = flow_for(protocol, 15.0)
    pred = predict(protocol, blood, background, flow, np.linspace(0, flow.path_mm(), 5),
                   background_structure_rel=ACQUISITION["background_structure_rel"].nominal)
    assert pred.blood.shape == (5, 1)
    assert np.all(np.isfinite(pred.cnr_efficiency))
    assert isinstance(pred, Prediction)
