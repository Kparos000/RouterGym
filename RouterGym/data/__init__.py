"""Data loaders for RouterGym."""

from __future__ import annotations

from typing import Any


def load_dataset(*args: Any, **kwargs: Any):
    from RouterGym.data.tickets import dataset_loader

    return dataset_loader.load_dataset(*args, **kwargs)


def load_kb(*args: Any, **kwargs: Any):
    from RouterGym.data.policy_kb import kb_loader

    return kb_loader.load_kb(*args, **kwargs)


__all__ = ["load_dataset", "load_kb"]
