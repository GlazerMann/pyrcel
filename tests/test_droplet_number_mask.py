"""Standalone regressions for this fix on unmodified upstream."""

import numpy as np
import pytest

from pyrcel import AerosolSpecies, ParcelModel
from pyrcel.activation._common import _kohler_crit_approx


def test_nd_excludes_unactivated_tail():
    a = AerosolSpecies("mix", {"r_drys": [0.05, 0.5], "Nis": [100.0, 200.0]}, 0.6)
    m = ParcelModel([a], 0.5, 283.15, -0.01, 85000.0)
    rc, _ = _kohler_crit_approx(283.15, a.r_drys, 0.6)
    m.time = np.array([0.0, 1.0])
    m.x = np.tile(m.y0, (2, 1))
    m.x[-1, 6] = 0.01
    m.x[-1, 7:] = [2 * float(rc[0]), a.r_drys[1]]
    s = m._compute_summary((0.01, 1.0))
    assert s["total_Nd"] == pytest.approx(100e6)
    assert s["total_nd_frac"] == pytest.approx(1 / 3)

    # Both bins meet the equilibrium criterion; only the first is a droplet.
    assert s["per_species"][0]["N_act"] == pytest.approx(300e6)
    assert s["total_Nd"] < s["per_species"][0]["N_act"]
