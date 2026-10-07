import pytest

from tissues import DERMIS, Confidence, Param, low_confidence_entries, low_confidence_report


def test_dermis_is_proton_density_reference():
    assert (DERMIS.pd_rel.lo, DERMIS.pd_rel.nominal, DERMIS.pd_rel.hi) == (1.0, 1.0, 1.0)


def test_report_lists_only_low_and_very_low_entries():
    entries = low_confidence_entries()
    assert entries
    assert all(p.confidence in (Confidence.LOW, Confidence.VERY_LOW) for _, _, p in entries)
    assert ("dermis", "t2star_ms", DERMIS.t2star_ms) in entries
    assert ("dermis", "pd_rel", DERMIS.pd_rel) not in entries
    assert len(low_confidence_report().splitlines()) == len(entries)


def test_param_rejects_nominal_outside_range():
    with pytest.raises(ValueError):
        Param(1.0, 2.0, 3.0, Confidence.LOW, "test")
