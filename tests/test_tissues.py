import numpy as np
import pytest

from tissues import (
    ALL_TISSUES,
    DERMIS,
    HAEMODYNAMICS,
    PARAM_FIELDS,
    TISSUES,
    Confidence,
    Param,
    TissueSample,
    collect_params,
    low_confidence_entries,
    low_confidence_report,
    sample,
    sample_tissues,
    signal_weighting,
)


def test_dermis_is_proton_density_reference():
    assert (DERMIS.pd_rel.lo, DERMIS.pd_rel.nominal, DERMIS.pd_rel.hi) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize("tissue", ALL_TISSUES, ids=lambda t: t.name)
def test_no_draw_can_have_t2star_above_t2(tissue):
    assert tissue.t2star_ms.hi <= tissue.t2_ms.lo


def test_report_lists_only_low_and_very_low_entries():
    entries = low_confidence_entries()
    assert all(p.confidence in (Confidence.LOW, Confidence.VERY_LOW) for _, _, p in entries)
    assert ("dermis", "t2star_ms", DERMIS.t2star_ms) in entries
    assert ("dermis", "pd_rel", DERMIS.pd_rel) not in entries
    assert ("haemodynamics", "target_vessel_velocity", HAEMODYNAMICS["target_vessel_velocity"]) in entries
    assert len(low_confidence_report().splitlines()) == len(entries)


def test_param_rejects_nominal_outside_range_and_log_without_positive_bound():
    with pytest.raises(ValueError):
        Param(1.0, 2.0, 3.0, Confidence.LOW, "test")
    with pytest.raises(ValueError):
        Param(0.0, 2.0, 1.0, Confidence.LOW, "test", log=True)


def test_lhs_draws_are_deterministic_stratified_and_inside_intervals():
    params = collect_params(["arterial_blood", "dermis"], haemodynamic_keys=("target_vessel_velocity",))
    a, b = sample(params, 64, seed=3), sample(params, 64, seed=3)
    for key, p in params.items():
        np.testing.assert_array_equal(a[key], b[key])
        assert p.lo <= a[key].min() and a[key].max() <= p.hi
    u = np.log(a["target_vessel_velocity"] / 10.0) / np.log(15.0)  # log-uniform -> one draw per stratum
    assert sorted(np.floor(u * 64).astype(int)) == list(range(64))
    assert not np.array_equal(sample(params, 64, seed=4)["dermis.t1_ms"], a["dermis.t1_ms"])


def test_sample_tissues_returns_tissue_samples_along_one_sample_axis():
    tissues, others = sample_tissues(["arterial_blood", "hypodermis"], 32, seed=0,
                                     haemodynamic_keys=("target_vessel_velocity",))
    blood = tissues["arterial_blood"]
    assert isinstance(blood, TissueSample) and blood.name == TISSUES["arterial_blood"].name
    assert all(np.shape(getattr(blood, f)) == (32,) for f in PARAM_FIELDS)
    assert set(others) == {"target_vessel_velocity"}
    assert blood.at(5).t1_ms == pytest.approx(blood.t1_ms[5])


def test_nominal_method_and_signal_weighting():
    tissues, _ = sample_tissues(["dermis"], 1, seed=0, method="nominal")
    assert tissues["dermis"].t2star_ms[0] == DERMIS.t2star_ms.nominal
    nominal = TissueSample.nominal(DERMIS)
    assert signal_weighting(nominal, 0.0) == pytest.approx(1.0)
    assert signal_weighting(nominal, 15.0) == pytest.approx(np.exp(-1.0))
