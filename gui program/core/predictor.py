"""Two-stage predictor of the GUI in plain numpy (no scikit-learn at run time).

Stage 1: the eight pushover quantities (V_y, delta_y, V_max, delta_max, V(delta_t), delta_t, delta_u, T_1), each the mean of
five networks that differ only in their random initialization (random_state 42, 1, 2, 3, 4). Stage 2: the five demand
ratios built from the stage-1 predictions and the soft vote of five classifier networks on the 21 inputs plus the five
ratios. The arrays in assets/models/two_stage_weights.npz are the fitted StandardScaler and MLP of every network
(gui/train_deployment_models.py; seeds, layouts and provenance in assets/models/model_metadata.json). The forward pass
repeats scikit-learn's operations in the same order (x - mean, / scale, then per layer x @ W + b with the hidden
activation in place, identity output for stage 1, softmax output for stage 2), so it returns the numbers of the fitted
pipelines.

As in the paper:
- inverse transforms V = exp(z) - 1 (V_y, V_max), delta = exp(z) H_tot (displacements), z unchanged (V(delta_t), T_1);
- the V(delta_t) of every network is capped at the V_max of the network with the same seed, then the five networks are
  averaged in physical units (seeds in ascending order);
- ratios theta_t = delta_t / H_tot, DCR_delta = delta_t / delta_u, mu_d = delta_t / delta_y, r_V = V(delta_t) / V_max,
  r_W = V_y / (W S_a(T_1)), with S_a from the design spectrum of the reference site at the predicted T_1;
- class probabilities: mean of the five softmax outputs, divided by their sum; the predicted state is the state of the
  largest probability (IO, LS, CP, C).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .frame import HAZARDS, INPUT_ORDER, FrameInput, derived, feature_vector, seismic_weight, spectral

MODEL_DIR = Path(__file__).resolve().parents[1] / 'assets' / 'models'
TARGETS = ['Vy', 'dy', 'Vmax', 'dmax', 'V_at_dt', 'dt', 'du', 'T1']
RATIOS = ['theta_t', 'DCR_delta', 'mu_d', 'V_at_dt_over_Vmax', 'Vy_over_W_Sa']
STATES = ['IO', 'LS', 'CP', 'C']
TL_SPECTRUM = 6.0        # s, long-period transition of the design spectrum
EPS = 1e-9


def sae_g(T, SDS, SD1, TL=TL_SPECTRUM):
    """Horizontal elastic design spectrum [g] (TBEC 2018 Eq. 2.2), as the classification notebook evaluates it."""
    TB = SD1 / SDS
    TA = 0.2 * TB
    T = max(float(T), 1e-6)
    if T <= TA:
        return SDS * (0.4 + 0.6 * T / TA)
    if T <= TB:
        return SDS
    if T <= TL:
        return SD1 / T
    return SD1 * TL / (T * T)


def sa_vector(T1, SDS, SD1):
    return np.array([sae_g(t, s, s1) for t, s, s1 in zip(T1, SDS, SD1)])


def demand_ratios(pred: dict, H, W, SDS, SD1) -> dict:
    """The five demand ratios from the stage-1 predictions (classification notebook, derive_augmented)."""
    Vy = np.maximum(pred['Vy'], EPS)
    Vmax = np.maximum(pred['Vmax'], EPS)
    Vat = np.asarray(pred['V_at_dt'], float)
    dy = np.maximum(pred['dy'], EPS)
    dt = np.maximum(pred['dt'], EPS)
    du = np.maximum(pred['du'], EPS)
    Sa = sa_vector(np.maximum(pred['T1'], EPS), SDS, SD1)
    H = np.maximum(H, EPS)
    return {'theta_t': dt / H, 'DCR_delta': dt / du, 'mu_d': dt / dy, 'V_at_dt_over_Vmax': Vat / Vmax,
            'Vy_over_W_Sa': Vy / (np.maximum(W, EPS) * np.maximum(Sa, EPS))}


def _forward(L: dict, X: np.ndarray) -> np.ndarray:
    a = np.array(X, dtype=np.float64, copy=True)
    a -= L['mean']
    a /= L['scale']
    n = len(L['W'])
    for i in range(n):
        a = a @ L['W'][i]
        a += L['b'][i]
        if i != n - 1:
            if L['act'] == 'tanh':
                np.tanh(a, out=a)
            elif L['act'] == 'relu':
                np.maximum(a, 0, out=a)
            else:
                raise ValueError(L['act'])
    if L['out'] == 'softmax':
        tmp = a - a.max(axis=1)[:, np.newaxis]
        np.exp(tmp, out=a)
        a /= a.sum(axis=1)[:, np.newaxis]
        return a
    return a.ravel()


class TwoStageModel:
    def __init__(self, model_dir: Path = MODEL_DIR):
        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / 'model_metadata.json').read_text(encoding='utf-8'))
        assert self.meta['input_order'] == INPUT_ORDER, 'input order of the models differs from the GUI'
        assert self.meta['classifier']['feature_order'] == INPUT_ORDER + RATIOS
        assert self.meta['classifier']['states'] == STATES
        arr = np.load(model_dir / 'two_stage_weights.npz')
        fs = self.meta['first_stage']
        self.fs_seeds = list(fs['seeds'])                        # fit order
        self.mean_order = list(fs['mean_order'])                 # ascending seeds (order of the mean)
        self.transform = {t: fs['targets'][t]['transform'] for t in TARGETS}
        self.fs = {t: {s: self._layers(arr, f'fs__{t}__s{s}', fs['targets'][t]['n_layers'],
                                       fs['targets'][t]['activation'], 'identity') for s in self.fs_seeds}
                   for t in TARGETS}
        cl = self.meta['classifier']
        self.clf_seeds = list(cl['seeds'])                       # member order of the soft vote
        self.clf = [self._layers(arr, f'clf__s{s}', cl['n_layers'], cl['activation'], 'softmax')
                    for s in self.clf_seeds]
        self.gate = self.meta.get('gate', {}).get('tier', 'not run')

    @staticmethod
    def _layers(arr, pre, n, act, out):
        return dict(mean=arr[f'{pre}__mean'], scale=arr[f'{pre}__scale'],
                    W=[arr[f'{pre}__W{k}'] for k in range(n)], b=[arr[f'{pre}__b{k}'] for k in range(n)],
                    act=act, out=out)

    def _inverse(self, t: str, z: np.ndarray, H: np.ndarray) -> np.ndarray:
        tt = self.transform[t]
        if tt == 'log1p':
            return np.expm1(z)
        if tt == 'lognormH':
            return np.exp(z) * H
        if tt in ('none', 'raw'):
            return z
        raise ValueError(tt)

    def first_stage_members(self, X: np.ndarray) -> dict:
        """seed -> target -> prediction in physical units (before the cap)."""
        X = np.asarray(X, dtype=np.float64)
        H = X[:, INPUT_ORDER.index('Htot')]
        return {s: {t: self._inverse(t, _forward(self.fs[t][s], X), H) for t in TARGETS} for s in self.fs_seeds}

    def first_stage(self, X: np.ndarray) -> tuple:
        """(pred, capped, members): the mean of the five networks per target with the per-seed V(delta_t) cap;
        capped marks the rows where the cap lowered at least one network."""
        mem = self.first_stage_members(X)
        seeds = self.mean_order
        pred = {t: np.mean([mem[s][t] for s in seeds], axis=0) for t in TARGETS}
        capped_rows = np.zeros(len(pred['Vy']), bool)
        capped = []
        for s in seeds:
            v, vm = mem[s]['V_at_dt'], mem[s]['Vmax']
            capped_rows |= v > vm
            capped.append(np.minimum(v, vm))
        pred['V_at_dt'] = np.mean(capped, axis=0)
        return pred, capped_rows, mem

    def classify(self, X_aug: np.ndarray) -> tuple:
        """(probabilities normalized to sum 1, index of the predicted state)."""
        X_aug = np.asarray(X_aug, dtype=np.float64)
        pr = np.mean([_forward(L, X_aug) for L in self.clf], axis=0)
        pr = pr / np.maximum(pr.sum(axis=1, keepdims=True), 1e-12)
        return pr, np.argmax(pr, axis=1)

    def predict_rows(self, X: np.ndarray, W, SDS, SD1) -> dict:
        """Full two-stage prediction for rows of the 21 inputs (paper order) with the seismic weight W [kN] and the
        spectral parameters of each row."""
        X = np.asarray(X, dtype=np.float64)
        pred, capped, mem = self.first_stage(X)
        rat = demand_ratios(pred, X[:, INPUT_ORDER.index('Htot')], np.asarray(W, float), np.asarray(SDS, float),
                            np.asarray(SD1, float))
        X_aug = np.column_stack([X] + [rat[k] for k in RATIOS])
        prob, cls = self.classify(X_aug)
        return dict(pred=pred, capped=capped, members=mem, ratios=rat, X_aug=X_aug, prob=prob, cls=cls)


@dataclass
class FramePrediction:
    frame: FrameInput
    derived: dict
    values: dict = field(default_factory=dict)      # hazard -> target -> value
    ratios: dict = field(default_factory=dict)      # hazard -> ratio -> value
    prob: dict = field(default_factory=dict)        # hazard -> state -> probability
    state: dict = field(default_factory=dict)       # hazard -> predicted state
    capped: dict = field(default_factory=dict)      # hazard -> bool
    extra: dict = field(default_factory=dict)       # W, S_DS, S_D1, S_a(T1) per hazard
    warnings: list = field(default_factory=list)


def predict_frame(model: TwoStageModel, f: FrameInput) -> FramePrediction:
    """Both hazard levels of one frame (the hazard level is not an input of the GUI)."""
    der = derived(f)
    X = np.array([feature_vector(f, h, der) for h in HAZARDS])
    W = np.array([seismic_weight(f)] * len(HAZARDS))
    sp = [spectral(f, h) for h in HAZARDS]
    SDS, SD1 = np.array([s[0] for s in sp]), np.array([s[1] for s in sp])
    r = model.predict_rows(X, W, SDS, SD1)
    out = FramePrediction(frame=f, derived=der)
    for i, h in enumerate(HAZARDS):
        out.values[h] = {t: float(r['pred'][t][i]) for t in TARGETS}
        out.ratios[h] = {k: float(r['ratios'][k][i]) for k in RATIOS}
        out.prob[h] = {s: float(r['prob'][i, j]) for j, s in enumerate(STATES)}
        out.state[h] = STATES[int(r['cls'][i])]
        out.capped[h] = bool(r['capped'][i])
        out.extra[h] = dict(W=float(W[i]), SDS=float(SDS[i]), SD1=float(SD1[i]),
                            Sa_T1=float(sae_g(max(out.values[h]['T1'], EPS), SDS[i], SD1[i])))
        v = out.values[h]
        if v['Vy'] > v['Vmax']:
            out.warnings.append(f'{h}: the predicted yield base shear exceeds the predicted peak base shear.')
        if out.capped[h]:
            out.warnings.append(f'{h}: the base shear at the target displacement was capped at the peak base shear in '
                                'at least one of the five networks.')
    return out
