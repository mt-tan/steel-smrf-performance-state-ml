"""Validity checks of the GUI: the Table 2 range check and the closed-form screens of the dataset generator.

Closed-form screens (generator_v021: filters.evaluate_all_pairs with the arguments of sampling.choose_sections, and the
height check of run_dataset), evaluated for the chosen column-beam pair with the tributary axial force of an interior
base column, Pu = (1.2 G + 1.6 Q) L_bay N_story:
  1. section ductility of the column (with Ca = Pu / (0.90 Fy A)) and of the beam, TBEC 2018 Table 9.3: high or limited
     ductility limits met (the database contains both levels; the level is shown);
  2. column axial-load ratio Pu / Pye <= 0.50, Pye = A Fye;
  3. gravity admissibility of the ground-storey column, Pg / Pc <= 0.50: Pg = (G (1 + 0.2 S_DS) + 0.30 Q) L_bay
     N_story (the DD-2 gravity state, G + 0.30 Q + 0.3 Ed(Z) with Ed(Z) = (2/3) S_DS G, S_DS of DD-2 and the soil
     class) times the interior-support coefficient of a continuous beam (1.25, 1.10, 1.143, 1.132 for 2 to 5 bays)
     times 1.03 for the self weight, Pc = Fcr A with Fye about the axis of the smaller radius of gyration,
     buckling length h1;
  4. beam gravity flexural check, w_u (L_bay - d_column)^2 / 12 <= 0.90 Wpl Fy with w_u = max(1.4 Gt, 1.2 Gt + 1.6 Q),
     Gt = G plus the beam self weight;
  5. clear lengths: L_bay - d_column > 0.20 m and h_typ - d_beam > 0.20 m;
  6. total height H_tot <= 28 m (height limit of the single-mode pushover at the reference site).
Fy is the nominal yield strength and Fye = 1.5 Fy (S235), 1.3 Fy (S275), 1.1 Fy (S355), TBEC 2018 Table 5.1.
Checks that need an analysis are listed and not evaluated: the first-mode effective mass ratio (>= 0.70) and the ratio
P/Pc <= 0.50 of every column under its gravity-state acceptance axial force (Annex 5C).
The strong-column-weak-beam, beam-depth and seismic strength-and-drift checks were not applied to the database and are
not checked here; the database frames are gravity-sized.
"""
from __future__ import annotations

import math

from .frame import CATALOGUE, E_STEEL, G_ACC, GRADES, RHO_STEEL, FrameInput, derived, spectral

MIN_CLEAR_LENGTH = 0.20      # m
H_MAX = 28.0                 # m
PHI_B, SPAN_DIVISOR = 0.90, 12.0
AXIAL_LIMIT = 0.50
SUPPORT_COEF = {2: 1.25, 3: 1.10, 4: 1.143, 5: 1.132}
EZ_COEF = 0.2               # 0.3 Ed(Z) / G = 0.3 (2/3) S_DS
NOT_CHECKED = [
    ('First-mode effective mass ratio', '>= 0.70, needs a modal analysis'),
    ('Column P/Pc under the acceptance axial force', '<= 0.50, needs the gravity analysis'),
]
NOTE = ('Preliminary screening inside the database domain; it does not replace a nonlinear analysis. Domain: planar '
        'gravity-sized frames without capacity design (not frames designed to current seismic provisions), 2 to 8 '
        'storeys, soil classes ZA to ZE, DD-2 and DD-1 spectra of the reference site only (no extrapolation to other '
        'spectra); pushover-based labels.')


def tributary_axial(f: FrameInput) -> float:
    """Interior base-column axial estimate under 1.2G + 1.6Q [kN] (generator filters.tributary_axial)."""
    w_u = 1.2 * f.G_line + 1.6 * f.Q_line
    trib = f.L_bay if f.n_bay >= 2 else f.L_bay / 2.0
    return w_u * trib * f.n_story


def ductility_level(p: dict, grade: str, Pu: float = 0.0) -> dict:
    """TBEC 2018 Table 9.3 at both levels (generator profiles.Section.ductility_level), nominal Fy."""
    Fy = GRADES[grade][0] * 1e3
    q = math.sqrt(E_STEEL / Fy)
    lam_f = p['bf'] / (2.0 * p['tf'])
    lam_w = (p['d'] - 2.0 * p['tf'] - 2.0 * p['r']) / p['tw']
    Ca = abs(Pu) / (0.90 * Fy * p['A'])
    lim_f_hd, lim_f_md = 0.30 * q, 0.38 * q
    if Ca <= 0.125:
        lim_w_hd = 2.45 * q * (1.0 - 0.93 * Ca)
        lim_w_md = 3.76 * q * (1.0 - 2.75 * Ca)
    else:
        lim_w_hd = max(0.77 * q * (2.93 - Ca), 1.49 * q)
        lim_w_md = max(1.12 * q * (2.33 - Ca), 1.49 * q)
    hd = (lam_f <= lim_f_hd) and (lam_w <= lim_w_hd)
    md = (lam_f <= lim_f_md) and (lam_w <= lim_w_md)
    return dict(level='high' if hd else ('limited' if md else 'none'), lam_f=lam_f, lam_w=lam_w, Ca=Ca,
                lim_f_high=lim_f_hd, lim_w_high=lim_w_hd, lim_f_limited=lim_f_md, lim_w_limited=lim_w_md)


def screens(f: FrameInput) -> list:
    """List of dicts: key, name, criterion, value, limit, passed."""
    c, b = CATALOGUE[f.col_profile], CATALOGUE[f.beam_profile]
    Fy_MPa, fye_factor = GRADES[f.grade]
    Fy = Fy_MPa * 1e3
    Fye = fye_factor * Fy
    Pu = tributary_axial(f)
    dc_col = ductility_level(c, f.grade, Pu)
    dc_beam = ductility_level(b, f.grade, 0.0)
    # axial-load ratio
    p_pye = abs(Pu) / (c['A'] * Fye)
    # gravity admissibility of the ground-storey column
    w_gq = (1.0 + EZ_COEF * spectral(f, 'DD-2')[0]) * f.G_line + 0.30 * f.Q_line     # DD-2 gravity state
    trib = f.L_bay if f.n_bay >= 2 else f.L_bay / 2.0
    alpha = SUPPORT_COEF.get(int(f.n_bay), 1.14) * 1.03
    P_g = abs(w_gq * trib * f.n_story) * alpha
    i_r = min(math.sqrt(c['I'] / c['A']), math.sqrt(c['Iz'] / c['A']))      # smaller principal-axis radius
    lam = f.h1 / i_r
    Fe = (math.pi ** 2) * E_STEEL / lam ** 2
    lam_c = 4.71 * math.sqrt(E_STEEL / Fye)
    Fcre = (0.658 ** (Fye / Fe)) * Fye if lam <= lam_c else 0.877 * Fe
    p_pc = P_g / (Fcre * c['A'])
    # beam gravity flexural check
    Gt = f.G_line + RHO_STEEL * G_ACC * b['A']                              # G with the beam self weight
    w_u = max(1.4 * Gt, 1.2 * Gt + 1.6 * f.Q_line)
    Mu = w_u * (f.L_bay - c['d']) ** 2 / SPAN_DIVISOR
    phiMp = PHI_B * b['Wpl'] * Fy
    # clear lengths and height
    clear_beam, clear_col = f.L_bay - c['d'], f.h_typ - b['d']
    Htot = derived(f)['Htot']
    return [
        dict(key='duct_column', name='Ductility, column', criterion='TBEC 2018 Table 9.3, high or limited',
             value=dc_col['level'], limit='high or limited', passed=dc_col['level'] != 'none', detail=dc_col),
        dict(key='duct_beam', name='Ductility, beam', criterion='TBEC 2018 Table 9.3, high or limited',
             value=dc_beam['level'], limit='high or limited', passed=dc_beam['level'] != 'none', detail=dc_beam),
        dict(key='axial', name='Column axial load', criterion='Pu/Pye under 1.2G + 1.6Q',
             value=f'{p_pye:.3f}', limit='<= 0.50', passed=bool(p_pye <= AXIAL_LIMIT)),
        dict(key='gravity_pc', name='Column gravity admissibility', criterion='Pg/Pc under G + 0.30Q + 0.3Ed(Z) of DD-2, ground storey, minimum-axis Pc',
             value=f'{p_pc:.3f}', limit='<= 0.50', passed=bool(p_pc <= AXIAL_LIMIT)),
        dict(key='beam_gravity', name='Beam gravity flexure', criterion='Mu <= 0.90 Wpl Fy under max(1.4G, 1.2G + 1.6Q), G with beam self weight',
             value=f'{Mu:.1f} kNm', limit=f'<= {phiMp:.1f} kNm', passed=bool(Mu <= phiMp)),
        dict(key='clear', name='Clear lengths', criterion='beam and column clear lengths above 0.20 m',
             value=f'{clear_beam:.2f} m, {clear_col:.2f} m', limit='> 0.20 m',
             passed=bool(clear_beam > MIN_CLEAR_LENGTH and clear_col > MIN_CLEAR_LENGTH)),
        dict(key='height', name='Total height', criterion='H_tot <= 28 m (single-mode pushover)',
             value=f'{Htot:.2f} m', limit='<= 28 m', passed=bool(Htot <= H_MAX)),
    ]


RANGE_KEYS = ['n_story', 'n_bay', 'L_bay', 'h1', 'h_typ', 'Htot', 'B_tot', 'G_line', 'Q_line', 'M_building', 'd_col',
              'bf_col', 'tf_col', 'tw_col', 'd_beam', 'bf_beam', 'tf_beam', 'tw_beam']


def range_check(f: FrameInput, realized: dict, der: dict | None = None) -> list:
    """Every numeric input against the realized ranges of the database (Table 2). List of dicts: key, value, lo, hi,
    inside."""
    der = der or derived(f)
    vals = dict(n_story=f.n_story, n_bay=f.n_bay, L_bay=f.L_bay, h1=f.h1, h_typ=f.h_typ, G_line=f.G_line,
                Q_line=f.Q_line, **der)
    out = []
    for k in RANGE_KEYS:
        lo, hi = realized[k]
        v = float(vals[k])
        out.append(dict(key=k, value=v, lo=lo, hi=hi, inside=bool(lo <= v <= hi)))
    return out
