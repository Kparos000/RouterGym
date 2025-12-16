from __future__ import annotations

import pytest

from RouterGym import label_space


def test_canonicalize_label_variants() -> None:
    assert label_space.canonicalize_label("HR Support") == "HR Support"
    assert label_space.canonicalize_label("hr_support") == "HR Support"
    assert label_space.canonicalize_label("  Admin Rights ") == "Administrative rights"
    assert label_space.canonicalize_label("vpn access issue") == "Access"
    assert label_space.canonicalize_label("Internal Project Work") == "Internal Project"
    assert label_space.canonicalize_label("storage quota increase") == "Storage"


def test_canonicalize_label_unknown_raises() -> None:
    with pytest.raises(RuntimeError):
        label_space.canonicalize_label("randomjunk")
