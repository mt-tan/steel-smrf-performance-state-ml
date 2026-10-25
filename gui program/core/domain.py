"""Joint-domain check of the GUI: is the combination of inputs of a frame represented in the database?

The range check (core/screens.py) tests each input alone; the sections of the database were chosen conditionally on the
frame variables, so inputs that are each inside their ranges can still form a combination the database does not contain.
This check measures the distance of the frame to its nearest training frame in the space of the 20 frame inputs (the 21
model inputs without the hazard level), each standardized with the mean and standard deviation of the training frames,
and compares it with the nearest-training-frame distances of the test frames of the paper (assets/domain_reference.npz,
written by make_domain_reference.py): the frame passes when its distance does not exceed their 99th percentile.
A pass does not validate a prediction; a fail flags a combination farther from the database than 99 percent of the
test frames.
"""
from __future__ import annotations

import json

import numpy as np

from .frame import ASSETS, FrameInput, derived, feature_vector

_REF = {}


def reference() -> dict:
    if not _REF:
        z = np.load(ASSETS / 'domain_reference.npz')
        meta = json.loads(str(z['meta']))
        _REF.update(Z=z['Z_train'].astype(np.float64), mean=z['mean'], std=z['std'], meta=meta,
                    limit=float(meta['percentiles_test_to_train']['99']),
                    p50=float(meta['percentiles_test_to_train']['50']))
    return _REF


def nearest_distance(f: FrameInput, der: dict | None = None) -> dict:
    """Distance of the frame to its nearest training frame (standardized frame inputs) and the limit of the check."""
    r = reference()
    x = np.asarray(feature_vector(f, 'DD-2', der)[:20], float)
    z = (x - r['mean']) / r['std']
    d = float(np.sqrt(np.min(np.sum((r['Z'] - z) ** 2, axis=1))))
    return dict(distance=d, limit=r['limit'], median_test=r['p50'], inside=bool(d <= r['limit']))


def check(f: FrameInput) -> dict:
    return nearest_distance(f, derived(f))
