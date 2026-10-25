"""Frame inputs and the derived model inputs of the GUI.

Pure-Python copies of the dataset generator (generator_v020, the generator of the database), no OpenSees import:
- section properties from assets/sections_catalogue.csv (byte copy of the generator catalogue; profiles.py);
- total height H_tot = h1 + (N_story - 1) h_typ and total width B_tot = N_bay L_bay;
- seismic mass M of the modelled frame: the nodal masses of model_builder.Frame2D.build (beam line load D + 0.3 L plus
  the beam self weight over the tributary width, and half the column self mass above and below each floor; no leaning
  column), summed in the same order;
- seismic weight W = (D + 0.3 L) B_tot N_story of the demand ratio r_W, as the classification notebook forms it;
- the spectral parameters S_DS and S_D1 of the reference site for the soil class and the hazard level (the values stored
  in the database, assets/gui_spec.json);
- the 21 model inputs in the order of the paper for one hazard level.
Expected yield strength (used by the validity checks): Fye = 1.5 Fy (S235), 1.3 Fy (S275), 1.1 Fy (S355), TBEC 2018
Table 5.1.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / 'assets'

E_STEEL = 2.0e8          # kN/m2
G_ACC = 9.81             # m/s2
RHO_STEEL = 7.85         # t/m3
LIVE_LOAD_N = 0.30       # live-load participation in the seismic mass and weight
GRADES = {'S235': (235.0, 1.5), 'S275': (275.0, 1.3), 'S355': (355.0, 1.1)}   # nominal Fy [MPa], Fye / Fy (Table 5.1)
GRADE_ORD = {'S235': 0, 'S275': 1, 'S355': 2}
SOIL_ORD = {'ZA': 0, 'ZB': 1, 'ZC': 2, 'ZD': 3, 'ZE': 4}
HAZARDS = ['DD-2', 'DD-1']
HAZARD_ORD = {'DD-2': 0, 'DD-1': 1}
INPUT_ORDER = ['n_story', 'n_bay', 'L_bay', 'h1', 'h_typ', 'Htot', 'B_tot', 'G_line', 'Q_line', 'M_building',
               'grade_ord', 'soil_ord', 'd_col', 'bf_col', 'tf_col', 'tw_col', 'd_beam', 'bf_beam', 'tf_beam', 'tw_beam',
               'hazard_ord']
_SPEC = {}


def spec() -> dict:
    """assets/gui_spec.json (written by make_gui_spec.py), read once."""
    if not _SPEC:
        _SPEC.update(json.loads((ASSETS / 'gui_spec.json').read_text(encoding='utf-8')))
    return _SPEC


def load_catalogue(path: Path = ASSETS / 'sections_catalogue.csv') -> dict:
    """name -> properties in SI units (m based), as the generator's profiles.py reads them."""
    cat = {}
    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            name = row['name'].strip()
            cat[name] = dict(name=name, family=row['family'].strip(),
                             d=float(row['h_mm']) / 1e3, bf=float(row['b_mm']) / 1e3, tw=float(row['tw_mm']) / 1e3,
                             tf=float(row['tf_mm']) / 1e3, r=float(row['r_mm']) / 1e3, A=float(row['A_cm2']) / 1e4,
                             I=float(row['Iy_cm4']) / 1e8, Iz=float(row['Iz_cm4']) / 1e8, Wpl=float(row['Wply_cm3']) / 1e6,
                             d_mm=float(row['h_mm']), bf_mm=float(row['b_mm']), tw_mm=float(row['tw_mm']),
                             tf_mm=float(row['tf_mm']))
    return cat


CATALOGUE = load_catalogue()


@dataclass
class FrameInput:
    """Raw inputs typed or chosen by the user."""
    n_story: int
    n_bay: int
    L_bay: float        # m
    h1: float           # m
    h_typ: float        # m
    G_line: float       # kN/m
    Q_line: float       # kN/m
    grade: str          # S235, S275, S355
    soil: str           # ZA to ZE
    col_profile: str    # HE profile
    beam_profile: str   # IPE profile

    def as_dict(self) -> dict:
        return asdict(self)


def total_height(f: FrameInput) -> float:
    return f.h1 + (f.n_story - 1) * f.h_typ


def total_width(f: FrameInput) -> float:
    return f.n_bay * f.L_bay


def seismic_mass(f: FrameInput) -> float:
    """Seismic mass M [t] of the modelled frame (generator model_builder, no leaning column)."""
    col, beam = CATALOGUE[f.col_profile], CATALOGUE[f.beam_profile]
    ns, nb, Lb = int(f.n_story), int(f.n_bay), float(f.L_bay)
    Z = [0.0] + [f.h1 + (j - 1) * f.h_typ for j in range(1, ns + 1)]
    w_gq = f.G_line + LIVE_LOAD_N * f.Q_line
    w_beam = w_gq + RHO_STEEL * G_ACC * beam['A']
    wsc = RHO_STEEL * col['A']
    floor_mass = [0.0] * (ns + 1)
    for j in range(1, ns + 1):
        hb = Z[j] - Z[j - 1]
        ha = (Z[j + 1] - Z[j]) if j < ns else 0.0
        for a in range(nb + 1):
            trib = (Lb / 2.0 if a in (0, nb) else Lb)
            floor_mass[j] += trib * w_beam / G_ACC + wsc * (hb + ha) / 2.0
    return sum(floor_mass)


def seismic_weight(f: FrameInput) -> float:
    """W [kN] of the demand ratio r_W: (D + 0.3 L) B_tot N_story, as the classification notebook forms it."""
    return (f.G_line + LIVE_LOAD_N * f.Q_line) * total_width(f) * f.n_story


def spectral(f: FrameInput, hazard: str) -> tuple:
    """(S_DS, S_D1) [g] of the reference site for the soil class and hazard level (database values)."""
    s = spec()['spectral'][f.soil][hazard]
    return float(s['SDS']), float(s['SD1'])


def section_dims(f: FrameInput) -> dict:
    """The eight section dimensions of the model inputs [mm]."""
    c, b = CATALOGUE[f.col_profile], CATALOGUE[f.beam_profile]
    return dict(d_col=c['d_mm'], bf_col=c['bf_mm'], tf_col=c['tf_mm'], tw_col=c['tw_mm'],
                d_beam=b['d_mm'], bf_beam=b['bf_mm'], tf_beam=b['tf_mm'], tw_beam=b['tw_mm'])


def derived(f: FrameInput) -> dict:
    """Derived inputs shown read-only in the GUI."""
    return dict(Htot=total_height(f), B_tot=total_width(f), M_building=seismic_mass(f), **section_dims(f))


def feature_vector(f: FrameInput, hazard: str, der: dict | None = None) -> list:
    """The 21 model inputs in the paper order for one hazard level (DD-2 or DD-1)."""
    der = der or derived(f)
    vals = dict(n_story=float(f.n_story), n_bay=float(f.n_bay), L_bay=float(f.L_bay), h1=float(f.h1),
                h_typ=float(f.h_typ), G_line=float(f.G_line), Q_line=float(f.Q_line),
                grade_ord=float(GRADE_ORD[f.grade]), soil_ord=float(SOIL_ORD[f.soil]),
                hazard_ord=float(HAZARD_ORD[hazard]), **{k: float(v) for k, v in der.items()})
    return [vals[k] for k in INPUT_ORDER]
